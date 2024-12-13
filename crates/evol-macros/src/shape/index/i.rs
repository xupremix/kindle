//TODO: REMOVE

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

// pub trait Index<S: Shape> {
//     const INDEX_CHECK: ();
//     type IndexShape: Shape;
//     fn indexes() -> &'static [u32];
// }
//
// impl<
//     const D0: usize,
//     const D1: usize,
//     const D2: usize,
//     const I_D0: usize,
//     const I_D1: usize,
// > Index<Rank3<D0, D1, D2>> for I2<I_D0, I_D1> {
//     const INDEX_CHECK: () = assert!(I_D0 < D0 && I_D1 < D1);
//     type IndexShape = Rank1<I_D2>;
//     fn indexes() -> &'static [u32] {
//         &[I_D0 as u32, I_D1 as u32]
//     }
// }

pub(crate) fn i(dims: usize, idents: &[Ident], at_tk: bool) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { evol::prelude:: }
    };

    let mut toks = vec![];

    (1..=dims).for_each(|i| {
        let new_shape;
        let shape_name = Ident::new(&format!("Rank{}", dims), Span::call_site());
        let const_idents_shape = idents
            .iter()
            .map(|i| quote! { const #i: usize })
            .collect::<Vec<_>>();
        let shape = quote! { #shape_name<#(#idents),*> };
        let i_name = Ident::new(&format!("I{}", i), Span::call_site());
        let i_idents = (0..i)
            .map(|i| Ident::new(&format!("I_D{}", i), Span::call_site()))
            .collect::<Vec<_>>();
        let i_const_idents = i_idents
            .iter()
            .map(|i| quote! { const #i: usize })
            .collect::<Vec<_>>();

        if i == dims {
            new_shape = quote! { Scalar };
        } else {
            let new_name = Ident::new(&format!("Rank{}", dims - i), Span::call_site());
            let new_idents = (i..dims).map(|i| &idents[i]);
            new_shape = quote! {
                #new_name<#(#new_idents),*>
            };
        }

        let assert_check = i_idents.iter().zip(idents.iter()).map(|(i, d)| {
            quote! { #i < #d && }
        });
        let indexes = i_idents
            .iter()
            .map(|i| quote! { #i as u32 })
            .collect::<Vec<_>>();

        toks.push(quote! {
            impl<
                #(#const_idents_shape,)*
                #(#i_const_idents),*
            > #path Index<#shape> for #i_name<#(#i_idents),*> {
                const INDEX_CHECK: () = assert!(#(#assert_check)* true, "All indexes must be less than the selected dimension");
                type IndexShape = #new_shape;
                fn indexes() -> &'static [u32] {
                    &[#(#indexes),*]
                }
            }
        });
    });

    quote! {
        #(#toks)*
    }
}
