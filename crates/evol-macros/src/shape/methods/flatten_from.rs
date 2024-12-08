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
> FlattenFrom<Rank4<D0, D1, D2, D3>, 2> for Rank3<D0, D1, FD> {
    const FLATTEN_CHECK: () = assert!(
        FD == D2 * D3,
        "Flatten check failed for dimensions 2 onwards"
    );
}

*/

pub(crate) fn flatten_from(
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

    (0..dims - 1).for_each(|i| {
        let flat_name = Ident::new(&format!("Rank{}", i + 1), Span::call_site());

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

        let const_dims = idents.iter().map(|i| {
            quote! {
                const #i: usize
            }
        });
        let const_dims = quote! {
            #(#const_dims,)*
            const FD: usize
        };

        let assert_msg = format!("Flatten check failed for dimension {} onwards", i);
        let assert_check = (i..dims).map(|k| {
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
            > #path FlattenFrom<#name<#(#idents),*>, #i> for #flat_name<#(#new_idents),*> {
                const FLATTEN_CHECK: () = assert!(#assert_check, #assert_msg);
            }
        });
    });

    quote! {
        #(#out)*
    }
}
