use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

// pub trait Indexer<S: Shape> {
//     type IndexShape: Shape;
//     fn indexes() -> &'static [usize];
// }
//
// pub struct I3<
//     const D0: usize,
//     const D1: usize,
//     const D2: usize,
// >;
//
// impl<
//     const D0: usize,
//     const D1: usize,
//     const D2: usize
// > Indexer<Rank3<D0, D1, D2>> for
//   I3<0, 1, 2>

pub(crate) fn indexer(dims: usize, idents: &[Ident], at_tk: bool) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { kindle::prelude:: }
    };

    let mut out = vec![];

    let i_name = Ident::new(&format!("I{}", dims), Span::call_site());
    out.push({
        let const_idents = idents
            .iter()
            .map(|i| quote! { const #i: usize })
            .collect::<Vec<_>>();
        quote! {
            pub struct #i_name<
                #(#const_idents),*
            >;
        }
    });

    let name = Ident::new(&format!("Rank{}", dims), Span::call_site());
    [true, false].into_iter().for_each(|keep| {
        (1..=dims).for_each(|i| {
            let const_idents = idents
                .iter()
                .map(|i| quote! { const #i: usize })
                .collect::<Vec<_>>();
            let i_name = Ident::new(&format!("I{}", i), Span::call_site());
            (0..dims).combinations(i).for_each(|p| {
                let shape = if keep {
                    let shape_idents = (0..dims)
                        .map(|i| {
                            if p.contains(&i) {
                                quote! { 1 }
                            } else {
                                let i = &idents[i];
                                quote! { #i }
                            }
                        })
                        .collect::<Vec<_>>();
                    quote! {
                        #name<#(#shape_idents),*>
                    }
                } else if dims == p.len() {
                    let shape = Ident::new("Scalar", Span::call_site());
                    quote! {
                        #shape
                    }
                } else {
                    let shape_name =
                        Ident::new(&format!("Rank{}", dims - p.len()), Span::call_site());
                    let shape_idents = idents
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !p.contains(i))
                        .map(|(_, i)| i);
                    quote! {
                        #shape_name<#(#shape_idents),*>
                    }
                };
                out.push(quote! {
                    impl<
                        #(#const_idents),*
                    > #path Indexer<#name<#(#idents),*>, #keep>
                    for #i_name<#(#p),*> {
                        type IndexShape = #shape;
                        fn indexes() -> &'static [usize] {
                            &[#(#p),*]
                        }
                    }
                });
            });
        });
    });

    quote! {
        #(#out)*
    }
}
