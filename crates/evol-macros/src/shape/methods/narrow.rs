use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

// pub trait Narrow<const DIM: usize, const START: usize, const LEN: usize, Dst: Shape>:
//     Shape
// {
//     const NARROW_CHECK: ();
// }
//
// impl<
//     const START: usize,
//     const LEN: usize,
//     const D0: usize,
//     const D1: usize,
//     const D2: usize,
// > Narrow<0, START, LEN, Rank3<LEN, D1, D2>> for Rank3<D0, D1, D2> {
//     cosnt NARROW_CHECK: () = assert!(START + LEN <= D0, "Invalid narrow");
// }

pub(crate) fn narrow(
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
        quote! { evol::prelude:: }
    };

    let const_idents = idents
        .iter()
        .map(|i| {
            quote! {
                const #i: usize
            }
        })
        .collect::<Vec<_>>();

    let mut toks = vec![];

    (0..dims).for_each(|i| {
        let dt = &idents[i];
        let mut new_idents = idents.to_vec();
        new_idents.insert(i, Ident::new("LEN", Span::call_site()));
        new_idents.remove(i + 1);

        toks.push(quote! {
            impl<
                const START: usize,
                const LEN: usize,
                #(#const_idents),*
            > #path Narrow<#i, START, LEN> for #name<#(#idents),*> {
                const NARROW_CHECK: () = {
                    assert!(
                        START + LEN <= #dt,
                        "Invalid narrow"
                    );
                };
                type NarrowShape = #name<#(#new_idents),*>;
            }
        });
    });

    quote! {
        #(#toks)*
    }
}
