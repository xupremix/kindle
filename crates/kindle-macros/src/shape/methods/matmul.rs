use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

/*
impl<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
> Matmul<Rank4<D0, D1, D3, D4>> for Rank4<D0, D1, D2, D3> {
    type MatmulShape = Rank4<D0, D1, D2, D4>;
}
*/

pub(crate) fn matmul(
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
        quote! { kindle::prelude:: }
    };

    let mut idents = idents.to_vec();
    idents.push(Ident::new(&format!("D{}", dims), Span::call_site()));

    let const_dims = idents
        .iter()
        .map(|i| {
            quote! {
                const #i: usize
            }
        })
        .collect::<Vec<_>>();

    let mut source_idents = idents.clone();
    source_idents.pop();

    let mut rhs_idents = idents.clone();
    rhs_idents.remove(dims - 2);

    let mut new_idents = idents.clone();
    new_idents.remove(dims - 1);

    quote! {
        impl<
            #(#const_dims),*
        > #path Matmul<#name<#(#source_idents),*>> for #name<#(#rhs_idents),*> {
            type MatmulShape = #name<#(#new_idents),*>;
        }
    }
}
