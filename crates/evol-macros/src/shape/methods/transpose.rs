use itertools::Itertools as _;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub(crate) fn transpose(
    dims: usize,
    name: &TokenStream,
    idents: &[Ident],
    at_tk: bool,
) -> TokenStream {
    if dims < 2 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { evol::prelude:: }
    };

    let const_dims = idents
        .iter()
        .map(|i| {
            quote! {
                const #i: usize
            }
        })
        .collect::<Vec<_>>();

    let mut out = vec![];

    (0..dims).combinations(2).for_each(|v| {
        let i = v[0];
        let j = v[1];

        let mut transposed = idents.to_vec();
        transposed.swap(i, j);

        out.push(quote! {
            impl<
                #(#const_dims),*
                > #path Transpose<#i, #j> for #name<#(#idents),*> {
                type Transposed = #name<#(#transposed),*>;
            }
        });
    });

    quote! {
        #(#out)*
    }
}
