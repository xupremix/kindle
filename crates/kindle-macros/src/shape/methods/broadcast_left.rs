use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn broadcast_left(dims: usize, name: &TokenStream, at_tk: bool) -> TokenStream {
    if dims < 1 {
        return quote! {};
    }

    let mut toks = vec![];

    let mut low = 1;
    let mut high = dims - 1;

    let path = if at_tk {
        quote! {}
    } else {
        quote! { kindle::prelude:: }
    };

    while low < dims {
        let low_name = Ident::new(&format!("Rank{}", low), Span::call_site());
        let high_name = Ident::new(&format!("Rank{}", high), Span::call_site());

        let low_const_dims = (0..low)
            .map(|i| {
                let i = Ident::new(&format!("L_D{}", i), Span::call_site());
                quote! { const #i: usize }
            })
            .collect::<Vec<_>>();
        let high_const_dims = (0..high)
            .map(|i| {
                let i = Ident::new(&format!("H_D{}", i), Span::call_site());
                quote! { const #i: usize }
            })
            .collect::<Vec<_>>();
        let const_dims = quote! {
            #(#low_const_dims,)*
            #(#high_const_dims,)*
        };

        let low_dims = (0..low)
            .map(|i| Ident::new(&format!("L_D{}", i), Span::call_site()))
            .collect::<Vec<_>>();
        let high_dims = (0..high)
            .map(|i| Ident::new(&format!("H_D{}", i), Span::call_site()))
            .collect::<Vec<_>>();

        toks.push(quote! {
            impl<
                #const_dims
            > #path BroadcastLeft< #high_name<#(#high_dims),*> >
            for #low_name<#(#low_dims),*> {
                type Extended = #name<#(#low_dims,)* #(#high_dims),*>;
            }
        });

        low += 1;
        high -= 1;
    }

    quote! {
        #(#toks)*
    }
}
