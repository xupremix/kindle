use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

// pub trait GetOnDim<const DIM: usize, const N: usize>: Shape {
//     const GET_ON_DIM_CHECK: ();
//     type GetOnDimShape: Shape;
// }
//
// impl<
//    const N: usize,
//    const D0: usize,
//    const D1: usize,
//    const D2: usize,
// > GetOnDim<0, N> for Rank3<D0, D1, D2> {
//    const GET_ON_DIM_CHECK: () = assert!(N < D0, "Index out of bounds");
//    type GetOnDimShape = Rank2<D1, D2>;
// }
//
// impl<
//    const N: usize,
//    const D0: usize,
//    const D1: usize,
//    const D2: usize,
// > GetOnDim<1, N> for Rank3<D0, D1, D2> {
//    const GET_ON_DIM_CHECK: () = assert!(N < D1, "Index out of bounds");
//    type GetOnDimShape = Rank2<D0, D2>;
// }

pub(crate) fn get_on_dim(
    dims: usize,
    name: &TokenStream,
    idents: &[Ident],
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

    let const_idents = idents
        .iter()
        .map(|i| quote! { const #i: usize })
        .collect::<Vec<_>>();

    let mut toks = vec![];

    (0..dims).for_each(|i| {
        let shape = if dims == 1 {
            quote! { Scalar }
        } else {
            let new_name = Ident::new(&format!("Rank{}", dims - 1), Span::call_site());
            let new_idents = (0..dims).filter(|&j| j != i).map(|i| &idents[i]);
            quote! { #new_name<#(#new_idents),*> }
        };

        let cmp_i = &idents[i];

        toks.push(quote! {
            impl<
                const N: usize,
                #(#const_idents),*
            > #path GetOnDim<#i, N> for #name<#(#idents),*> {
                const GET_ON_DIM_CHECK: () = assert!(N < #cmp_i, "Index out of bounds");
                type GetOnDimShape = #shape;
            }
        });
    });

    quote! {
        #(#toks)*
    }
}
