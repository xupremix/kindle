use arg::Args;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::Ident;

mod arg;
mod index;
mod methods;

pub(crate) fn shape(input: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(input as Args);
    let vis = args.vis;
    let dims = args.dims;
    let at_tk = args.at_tk;

    let name = if dims == 0 {
        quote! { Scalar }
    } else {
        let i = Ident::new(&format!("Rank{}", dims), Span::call_site());
        quote! { #i }
    };

    let idents = (0..dims)
        .map(|i| Ident::new(&format!("D{}", i), Span::call_site()))
        .collect::<Vec<_>>();

    let const_idents = idents
        .iter()
        .map(|i| quote! { const #i: usize })
        .collect::<Vec<_>>();

    let nelems = if dims == 0 {
        quote! { 1 }
    } else {
        quote! { 1 #(* #idents)* }
    };

    let const_generics = if dims == 0 {
        quote! {}
    } else {
        quote! { <#(#const_idents),*> }
    };

    let generics = if dims == 0 {
        quote! {}
    } else {
        quote! { <#(#idents),*> }
    };

    let segment = if at_tk {
        quote! {
            crate::shape::
        }
    } else {
        quote! {
            kindle::shape::
        }
    };

    let shape = if dims == 0 {
        quote! { [K; 0] }
    } else {
        gen_shape(dims - 1, 0)
    };

    let candle_shape_segment = if at_tk {
        quote! {}
    } else {
        quote! {
            kindle::prelude::
        }
    };

    let as_slice = if dims <= 1 {
        quote! { shape.as_slice() }
    } else {
        let mut out = vec![];
        out.push(quote! { shape.as_flattened() });
        for _ in 0..dims - 2 {
            out.push(quote! { .as_flattened() });
        }
        quote! { #(#out)* }
    };

    let methods = methods::methods(dims, &name, &idents, at_tk);
    let index = index::index(dims, &idents, at_tk);

    quote! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #vis struct #name #const_generics;

        impl #const_generics #segment Shape for #name #generics {
            type Shape<K: #candle_shape_segment Kind> = #shape;
            const DIMS: usize = #dims;
            const NELEMS: usize = #nelems;

            fn shape() -> #candle_shape_segment CandleShape {
                #candle_shape_segment CandleShape::from_dims(&[#(#idents),*])
            }

            fn dims() -> &'static [usize] {
                &[#(#idents),*]
            }

            #[inline(always)]
            fn as_slice<K: Kind>(shape: &Self::Shape<K>) -> &[K] {
                #as_slice
            }
        }

        #methods

        #index

    }
    .into()
}

fn gen_shape(dims: usize, curr: usize) -> proc_macro2::TokenStream {
    let dim = Ident::new(&format!("D{}", curr), proc_macro2::Span::call_site());
    if dims == 0 {
        return quote! { [K; #dim] };
    }
    let shape = gen_shape(dims - 1, curr + 1);
    quote! {
        [#shape; #dim]
    }
}
