use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

// pub trait Stack<const DIM: usize, const N: usize>: Shape {
//     type StackShape: Shape;
// }
//
// impl<
//     const N: usize,
//     const D0: usize,
//     const D1: usize,
//     const D2: usize,
// > Stack<0, N> for Rank3<D0, D1, D2> {
//     type StackShape = Rank4<N, D0, D1, D2>;
// }
// > Stack<1, N> for Rank3<D0, D1, D2> {
//     type StackShape = Rank4<D0, N, D1, D2>;
// }
// > Stack<2, N> for Rank3<D0, D1, D2> {
//    type StackShape = Rank4<D0, D1, N, D2>;
// }
// > Stack<3, N> for Rank3<D0, D1, D2> {
//    type StackShape = Rank4<D0, D1, D2, N>;
// }
//

pub(crate) fn stack(dims: usize, name: &TokenStream, idents: &[Ident], at_tk: bool) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { kindle::prelude:: }
    };

    let mut idents = idents.to_vec();
    idents.pop();

    let const_idents = idents
        .iter()
        .map(|i| quote! { const #i: usize })
        .collect::<Vec<_>>();

    let from = if dims == 1 {
        quote! { Scalar }
    } else {
        let name = Ident::new(&format!("Rank{}", dims - 1), Span::call_site());
        quote! { #name<#(#idents),*> }
    };

    let mut toks = vec![];

    (0..dims).for_each(|i| {
        let to = {
            idents.insert(i, Ident::new("N", Span::call_site()));
            quote! { #name<#(#idents),*> }
        };
        idents.remove(i);

        toks.push(quote! {
            impl<
                const N: usize,
                #(#const_idents),*
            > #path Stack<#i, N> for #from {
                type StackShape = #to;
            }
        });
    });

    quote! {
        #(#toks)*
    }
}
