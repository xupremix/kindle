use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub(crate) fn t(dims: usize, name: &TokenStream, idents: &[Ident], at_tk: bool) -> TokenStream {
    if dims < 2 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { evol::prelude:: }
    };

    let const_dims = idents.iter().map(|i| {
        quote! {
            const #i: usize
        }
    });

    let mut transposed = idents.to_vec();
    transposed.swap(idents.len() - 1, idents.len() - 2);

    quote! {
        impl<
            #(#const_dims),*
        > #path T for #name<#(#idents),*> {
            type Transposed = #name<#(#transposed),*>;
        }

    }
}
