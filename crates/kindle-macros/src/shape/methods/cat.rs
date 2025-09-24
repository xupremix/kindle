use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

// pub trait Cat<const DIM: usize, const N: usize, Dst: Shape>: Shape {
//     const CAT_CHECK: ();
// }
//
// impl<
//     const T: usize,
//     const N: usize,
//     const D0: usize,
//     const D1: usize,
//     const D2: usize,
// > Cat<0, N, Rank3<T, D1, D2>> for Rank3<D0, D1, D2> {
//     const CAT_CHECK: () = assert!(T == D0 * N);
// }
// > Cat<1, N, Rank3<D0, T, D2>> for Rank3<D0, D1, D2> {
//     const CAT_CHECK: () = assert!(T == D1 * N);
// }
// ...

pub(crate) fn cat(dims: usize, name: &TokenStream, idents: &[Ident], at_tk: bool) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { kindle::prelude:: }
    };

    let const_idents = idents
        .iter()
        .map(|i| quote! { const #i: usize })
        .collect::<Vec<_>>();

    let mut toks = vec![];

    (0..dims).for_each(|i| {
        let t = &idents[i];
        let assert_msg = format!("Cat check failed for dimension {}", i);
        let mut new_idents = idents.to_vec();
        new_idents.insert(i, Ident::new("ND", Span::call_site()));
        new_idents.remove(i + 1);

        toks.push(quote! {
            impl<
                const ND: usize,
                const N: usize,
                #(#const_idents),*
            > #path Cat<#i, N, #name<#(#new_idents),*>> for #name<#(#idents),*> {
                const CAT_CHECK: () = assert!(ND == #t * N, #assert_msg);
            }
        });
    });

    quote! {
        #(#toks)*
    }
}
