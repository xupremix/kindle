use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub(crate) fn swiglu(
    dims: usize,
    name: &TokenStream,
    idents: &[Ident],
    at_tk: bool,
) -> TokenStream {
    if dims < 1 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { kindle::prelude:: }
    };

    let mut const_idents = idents
        .iter()
        .map(|i| {
            quote! {
                const #i: usize
            }
        })
        .collect::<Vec<_>>();
    const_idents.pop();

    let mut idents = idents.to_vec();
    idents.pop();

    quote! {
        impl<
            #(#const_idents),*
        > #path SwigluShape for #name<#(#idents,)* 2> {
            type SwigluShape = #name<#(#idents,)* 1>;
        }
    }
}
