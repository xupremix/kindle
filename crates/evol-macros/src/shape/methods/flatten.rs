use itertools::Itertools as _;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

/*

impl<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const FD: usize,
> Flatten<Rank4<D0, D1, D2, D3>, 1, 2> for Rank3<D0, FD, D3> {
    const FLATTEN_CHECK: () = assert!(
        FD == D1 * D2,
        "Flatten check failed for dimensions 1 and 2"
    );
}

*/

pub(crate) fn flatten(
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
        quote! { evol::prelude:: }
    };

    let mut out = vec![];

    (0..dims).combinations(2).for_each(|v| {
        let i = v[0];
        let j = v[1];

        let flat_name = Ident::new(&format!("Rank{}", dims - (j - i)), Span::call_site());

        let mut new_idents = (0..i)
            .map(|i| {
                let i = &idents[i];
                quote! {
                    #i
                }
            })
            .collect::<Vec<_>>();
        new_idents.push(quote! {
            FD
        });
        new_idents.extend((j + 1..dims).map(|i| {
            let i = &idents[i];
            quote! {
                #i
            }
        }));

        let const_dims = idents.iter().map(|i| {
            quote! {
                const #i: usize
            }
        });
        let const_dims = quote! {
            #(#const_dims,)*
            const FD: usize
        };

        let assert_msg = format!("Flatten check failed for dimensions {} and {}", i, j);
        let assert_check = (i..j + 1).map(|k| {
            let i = &idents[k];
            quote! {
                #i *
            }
        });
        let assert_check = quote! {
            FD == #(#assert_check)* 1
        };

        out.push(quote! {
            impl<
                #const_dims
            > #path Flatten<#name<#(#idents),*>, #i, #j> for #flat_name<#(#new_idents),*> {
                const FLATTEN_CHECK: () = assert!(#assert_check, #assert_msg);
            }
        });
    });

    quote! {
        #(#out)*
    }
}
