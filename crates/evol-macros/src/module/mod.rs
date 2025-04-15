use phf::{phf_map, phf_set};
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, TokenStreamExt};
use std::collections::VecDeque;
use syn::{parse_macro_input, Data, Ident, LitInt, Type};

#[derive(Debug)]
enum Layers {
    Normal,
    Ignore,
    Swiglu,
}

static LAYERS: phf::Map<&'static str, Layers> = phf_map! {
    "Linear" => Layers::Normal,
    "Conv2d" => Layers::Normal,
    "Relu" => Layers::Ignore,
    "Swiglu" => Layers::Swiglu,
};

#[cfg(feature = "half")]
static TYPES: phf::Set<&'static str> = phf_set! {
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "f32",
    "f64",
    "f16",
    "bf16",
};

#[cfg(not(feature = "half"))]
static TYPES: phf::Set<&'static str> = phf_set! {
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "f32",
    "f64",
};

static DEVICES: phf::Set<&'static str> = phf_set! {
    "Cpu",
    "Cuda",
    "Metal",
};

struct LayerType {
    ty: Layers,
    dims: Vec<syn::LitInt>,
}

pub(crate) fn module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let module_impl = match input.data {
        Data::Struct(ref data) => {
            let name = &input.ident;
            let generics = &input.generics;
            let params = generics
                .type_params()
                .map(|param| param.ident.clone())
                .collect::<Vec<_>>();
            let mut contains_device = None;
            let mut contains_kind = None;
            for generic in generics.type_params() {
                for bound in generic.bounds.iter() {
                    if let syn::TypeParamBound::Trait(trait_bound) = bound {
                        let name = trait_bound.path.segments.last().unwrap().ident.to_string();
                        if name == "Kind" {
                            contains_kind = Some(generic.ident.clone());
                        } else if name == "Device" {
                            contains_device = Some(generic.ident.clone());
                        }
                    }
                }
            }
            let mut k = None;
            let mut d = None;
            if let Some(ref kind) = contains_kind {
                k = Some(quote! {
                    #kind : evol::prelude::Kind,
                });
            }
            if let Some(ref device) = contains_device {
                d = Some(quote! {
                    #device : evol::prelude::Device,
                });
            }
            let mut out = quote! {
                impl<
                    EvolShape: evol::prelude::Shape,
                    #k
                    #d
                > evol::prelude::Module<
                    evol::prelude::Tensor<
                        EvolShape,
            };
            let mut tmp = quote! {};
            tmp.append_all(quote! {
                    >
                > for #name <#(#params),*>
                where
            });
            let mut layer_types = VecDeque::new();
            let mut field_names = vec![];
            let mut found_type = None;
            let mut found_device = None;
            for field in data.fields.iter() {
                if let Type::Path(ref path) = field.ty {
                    let layer = path.path.segments.iter().last().unwrap();
                    if let Some(segment) = LAYERS.get(&layer.ident.to_string()) {
                        match segment {
                            Layers::Normal => match layer.arguments {
                                syn::PathArguments::AngleBracketed(ref angle_generics) => {
                                    let mut dims = vec![];
                                    for arg in angle_generics.args.iter() {
                                        if let syn::GenericArgument::Const(syn::Expr::Lit(
                                            ref lit,
                                        )) = arg
                                        {
                                            if let syn::Lit::Int(ref lit_int) = lit.lit {
                                                dims.push(lit_int.clone());
                                            }
                                        } else if let syn::GenericArgument::Type(Type::Path(
                                            ref p,
                                        )) = arg
                                        {
                                            let segment = p.path.segments.iter().last().unwrap();
                                            let type_name = segment.ident.to_string();
                                            if TYPES.contains(&type_name) && found_type.is_none() {
                                                found_type = Some(type_name);
                                            } else if DEVICES.contains(&type_name)
                                                && found_device.is_none()
                                            {
                                                let mut const_args = None;
                                                if let syn::PathArguments::AngleBracketed(
                                                    ref args,
                                                ) = segment.arguments
                                                {
                                                    if let syn::GenericArgument::Const(
                                                        syn::Expr::Lit(ref lit),
                                                    ) = args.args.iter().next().unwrap()
                                                    {
                                                        if let syn::Lit::Int(ref lit_int) = lit.lit
                                                        {
                                                            const_args = Some(lit_int.clone());
                                                        }
                                                    }
                                                }
                                                found_device = Some((type_name, const_args));
                                            }
                                        }
                                    }
                                    layer_types.push_front(LayerType {
                                        ty: Layers::Normal,
                                        dims: dims.clone(),
                                    });
                                    tmp.append_all(gen_where_clause(0, &layer_types));
                                    field_names.push(field.ident.clone().unwrap());
                                }
                                _ => panic!("Unsupported layer type: {:#?}", layer),
                            },
                            Layers::Swiglu => {
                                layer_types.push_front(LayerType {
                                    ty: Layers::Swiglu,
                                    dims: vec![],
                                });
                                tmp.append_all(gen_where_clause(0, &layer_types));
                                field_names.push(field.ident.clone().unwrap());
                            }
                            Layers::Ignore => field_names.push(field.ident.clone().unwrap()),
                        }
                    }
                }
            }
            tmp.append_all(gen_body(
                &layer_types,
                &field_names,
                &contains_kind,
                &contains_device,
                &found_type,
                &found_device,
            ));
            if let Some(found) = found_type {
                let i = Ident::new(&found, Span::call_site());
                out.append_all(quote! {
                    #i,
                });
            }
            if let Some(kind) = contains_kind {
                out.append_all(quote! {
                    #kind,
                });
            }
            if let Some((name, args)) = found_device {
                let i = Ident::new(&name, Span::call_site());
                out.append_all(quote! {
                    #i
                });
                if let Some(arg) = args {
                    out.append_all(quote! {
                        < #arg >
                    });
                }
                out.append_all(quote! { , });
            }
            if let Some(dev) = contains_device {
                out.append_all(quote! {
                    #dev,
                });
            }
            out.append_all(tmp);
            out
        }
        _ => panic!("The module macro can only be used on structs"),
    };
    quote! {
        #module_impl
    }
    .into()
}

fn gen_body(
    layer_types: &VecDeque<LayerType>,
    field_names: &[Ident],
    contains_kind: &Option<Ident>,
    contains_device: &Option<Ident>,
    found_type: &Option<String>,
    found_device: &Option<(String, Option<LitInt>)>,
) -> proc_macro2::TokenStream {
    let tensor_shape = gen_tensor_shape(0, layer_types);
    let first = &field_names[0];
    let first = quote! {
        let xs = self.#first.forward(xs);
    };
    let forward = field_names.iter().skip(1).map(|name| {
        quote! {
            let xs = self.#name.forward(&xs);
        }
    });
    let mut k = None;
    let mut d = None;
    if let Some(kind) = contains_kind {
        k = Some(quote! {
            #kind,
        });
    }
    let mut ty = None;
    if let Some(found) = found_type {
        let i = Ident::new(found, Span::call_site());
        ty = Some(quote! {
            #i,
        });
    }
    if let Some(device) = contains_device {
        d = Some(quote! {
            #device,
        });
    }
    let mut device = None;
    if let Some((name, args)) = found_device {
        let i = Ident::new(name, Span::call_site());
        device = if let Some(arg) = args {
            Some(quote! {
                #i < #arg >,
            })
        } else {
            Some(quote! {
                #i,
            })
        };
    }
    quote! {{
        type Output = evol::prelude::Tensor<
            #tensor_shape,
            #ty
            #k
            #d
            #device
        >;

        fn forward(&self, xs: &evol::prelude::Tensor<EvolShape, #ty #k #d #device>) -> Self::Output {
            #first
            #(#forward)*
            xs
        }
    }}
}

fn gen_tensor_shape(start: usize, layer_types: &VecDeque<LayerType>) -> proc_macro2::TokenStream {
    if start == layer_types.len() {
        quote! { EvolShape }
    } else {
        let middle = gen_where_clause(start + 1, layer_types);
        let dims = &layer_types[start].dims;
        match layer_types[start].ty {
            Layers::Normal => quote! {
                <#middle as evol::prelude::Forward<#(#dims),*>>::ForwardShape
            },
            Layers::Swiglu => quote! {
                <#middle as evol::prelude::SwigluShape>::SwigluShape
            },
            Layers::Ignore => unreachable!("An `Ignore` layer should never be added"),
        }
    }
}

fn gen_where_clause(start: usize, layer_types: &VecDeque<LayerType>) -> proc_macro2::TokenStream {
    if start == layer_types.len() {
        quote! { EvolShape }
    } else if start == 0 {
        let middle = gen_where_clause(start + 1, layer_types);
        let dims = &layer_types[start].dims;
        match layer_types[start].ty {
            Layers::Normal => quote! {
                #middle: evol::prelude::Forward<#(#dims),*>,
            },
            Layers::Swiglu => quote! {
                #middle: evol::prelude::SwigluShape,
            },
            Layers::Ignore => unreachable!("An `Ignore` layer should never be added"),
        }
    } else {
        let middle = gen_where_clause(start + 1, layer_types);
        let dims = &layer_types[start].dims;
        match layer_types[start].ty {
            Layers::Normal => quote! {
                <#middle as evol::prelude::Forward<#(#dims),*>>::ForwardShape
            },
            Layers::Swiglu => quote! {
                <#middle as evol::prelude::SwigluShape>::SwigluShape
            },
            Layers::Ignore => unreachable!("An `Ignore` layer should never be added"),
        }
    }
}
