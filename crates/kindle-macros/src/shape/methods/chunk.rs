use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

// pub trait Chunk<const DIM: usize, const NELEMS: usize>: Shape {
//     const CHUNK_CHECK: ();
//     type ChunkShape: Shape;
// }
//
// impl<
//     const NELEMS: usize,
//     const D0: usize,
//     const D1: usize,
//     const D2: usize,
// > Chunk<0, NELEMS> for Rank3<D0, D1, D2> {
//     const CHUNK_CHECK: () = assert!(D0 % NELEMS == 0 && NELEMS > 0, "Invalid chunk");
//     type ChunkShape = Rank3<NELEMS, D1, D2>;
// }
// > Chunk<1, NELEMS> for Rank3<D0, D1, D2> {
//     const CHUNK_CHECK: () = assert!(D1 % NELEMS == 0 && NELEMS > 0, "Invalid chunk");
//     type ChunkShape = Rank3<D0, NELEMS, D2>;
// }
// > Chunk<2, NELEMS> for Rank3<D0, D1, D2> {
//     const CHUNK_CHECK: () = assert!(D2 % NELEMS == 0 && NELEMS > 0, "Invalid chunk");
//     type ChunkShape = Rank3<D0, D1, NELEMS>;
// }

pub(crate) fn chunk(dims: usize, name: &TokenStream, idents: &[Ident], at_tk: bool) -> TokenStream {
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
        new_idents.insert(i, Ident::new("NELEMS", Span::call_site()));
        new_idents.remove(i + 1);

        toks.push(quote! {
            impl<
                const NELEMS: usize,
                #(#const_idents),*
            > #path Chunk<#i, NELEMS> for #name<#(#idents),*> {
                const CHUNK_CHECK: () = assert!(
                    #dt % NELEMS == 0 &&
                    NELEMS > 0,
                    "Invalid chunk"
                );
                type ChunkShape = #name<#(#new_idents),*>;
            }
        });
    });

    quote! {
        #(#toks)*
    }
}
