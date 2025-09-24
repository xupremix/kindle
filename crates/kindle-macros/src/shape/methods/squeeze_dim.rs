use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn squeeze_dim(
    dims: usize,
    name: &TokenStream,
    dim_idents: &[Ident],
    at_tk: bool,
) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { kindle::prelude:: }
    };

    if dims == 1 {
        return quote! {
            impl #path SqueezeDim<0>
                for Rank1<1> {
                type SqueezeShape = Scalar;
            }
        };
    }

    let mut out = vec![];

    for i in 0..dims {
        let mut bits = vec![false; dims];
        bits[i] = true;

        let new_name = Ident::new(&format!("Rank{}", dims - 1), Span::call_site());
        let mut count = 0;

        let const_dims = bits
            .iter()
            .filter_map(|&b| {
                if b {
                    None
                } else {
                    let i = &dim_idents[count];
                    count += 1;
                    Some(quote! {
                        const #i: usize
                    })
                }
            })
            .collect::<Vec<_>>();
        count = 0;
        let idents = bits
            .iter()
            .map(|&b| {
                if b {
                    quote! {
                        1usize
                    }
                } else {
                    let i = &dim_idents[count];
                    count += 1;
                    quote! {
                        #i
                    }
                }
            })
            .collect::<Vec<_>>();
        count = 0;
        let new_idents = bits
            .iter()
            .filter_map(|&b| {
                if b {
                    None
                } else {
                    let i = &dim_idents[count];
                    count += 1;
                    Some(quote! {
                        #i
                    })
                }
            })
            .collect::<Vec<_>>();

        out.push(quote! {
            impl<
               #(#const_dims),*
            > #path SqueezeDim<#i>
            for #name<#(#idents),*> {
                type SqueezeShape = #path #new_name<#(#new_idents),*>;
            }
        });
    }

    quote! {
        #(#out)*
    }
}
