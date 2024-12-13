use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn get(dims: usize, name: &TokenStream, idents: &[Ident], at_tk: bool) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { evol::prelude:: }
    };

    let const_idents = idents
        .iter()
        .map(|i| quote! { const #i: usize })
        .collect::<Vec<_>>();

    let shape = if dims == 1 {
        quote! { Scalar }
    } else {
        let new_name = Ident::new(&format!("Rank{}", dims - 1), Span::call_site());
        let new_idents = (1..dims).map(|i| &idents[i]);
        quote! { #new_name<#(#new_idents),*> }
    };

    quote! {
        impl<
            const N: usize,
            #(#const_idents),*
        > #path Get<N> for #name<#(#idents),*> {
            const GET_CHECK: () = assert!(N < D0, "Index out of bounds");
            type GetShape = #shape;
        }
    }
}
