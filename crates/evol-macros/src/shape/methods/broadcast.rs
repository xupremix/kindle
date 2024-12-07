use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn broadcast(dims: usize, name: &TokenStream, at_tk: bool) -> TokenStream {
    if dims < 1 {
        return quote! {};
    }

    let mut toks = vec![];
    let mut curr_dim = 1;

    let path = if at_tk {
        quote! {}
    } else {
        quote! { evol::prelude:: }
    };

    while curr_dim <= dims {
        let shape_curr = Ident::new(&format!("Rank{}", curr_dim), Span::call_site());
        let shape_gen = name;

        // Generation of const dims
        let const_dims = (0..curr_dim)
            .map(|i| {
                let i = Ident::new(&format!("D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();
        let const_dims_g = (0..dims)
            .map(|i| {
                let i = Ident::new(&format!("G_D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();
        let const_dims_d = (0..dims)
            .map(|i| {
                let i = Ident::new(&format!("D_D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();
        let const_dims = quote! {
            #(#const_dims)*
            #(#const_dims_g)*
            #(#const_dims_d)*
        };

        // Generation of idents
        let idents = (0..curr_dim)
            .map(|i| Ident::new(&format!("D{}", i), Span::call_site()))
            .collect::<Vec<_>>();
        let idents_g = (0..dims)
            .map(|i| Ident::new(&format!("G_D{}", i), Span::call_site()))
            .collect::<Vec<_>>();
        let idents_d = (0..dims)
            .map(|i| Ident::new(&format!("D_D{}", i), Span::call_site()))
            .collect::<Vec<_>>();

        let assert_msg = format!(
            "\nThe dimension provided for broadcasting {shape_curr} into {shape_gen} are not compatible.\nTo Broadcast when iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be:\n - equal\n - one of them is 1\n - one of them does not exist\n"
        );

        let assert_check = gen_assert_check(curr_dim, dims);

        toks.push(quote! {
            impl<
                #const_dims
                > #path Broadcast<
                    #shape_curr < #(#idents),* >,
                    #shape_gen < #(#idents_d),* >
                > for #shape_gen < #(#idents_g),* > {
                const BROADCAST_CHECK: () = assert!(#assert_check, #assert_msg);
            }
        });
        if curr_dim != dims {
            toks.push(quote! {
                impl<
                    #const_dims
                > #path Broadcast<
                        #shape_gen < #(#idents_g),* >,
                        #shape_gen < #(#idents_d),* >
                    > for #shape_curr < #(#idents),* > {
                    const BROADCAST_CHECK: () = assert!(#assert_check, #assert_msg);
                }
            });
        }
        curr_dim += 1;
    }
    quote! {
        #(#toks)*
    }
}

fn gen_assert_check(mut curr_dim: usize, mut dims: usize) -> TokenStream {
    // Reminder:
    // broadcast requirements:
    // When iterating over the dimension sizes, starting at the trailing dimension the dimension sizes must either be:
    // - equal,
    // - one of them is 1
    // - or one of them does not exist.
    let mut toks = vec![];
    while dims > 0 {
        let dim_g = Ident::new(&format!("G_D{}", dims - 1), Span::call_site());
        let dim_d = Ident::new(&format!("D_D{}", dims - 1), Span::call_site());
        if curr_dim == 0 && dims > 0 {
            toks.push(quote! {
                #dim_d == #dim_g &&
            });
            dims -= 1;
            continue;
        }
        let dim = Ident::new(&format!("D{}", curr_dim - 1), Span::call_site());
        toks.push(quote! {
            (#dim == #dim_g || #dim == 1 || #dim_g == 1 ) &&
            (#dim_d >= #dim_g && #dim_d >= #dim) &&
            (#dim_d == #dim_g || #dim_d == #dim) &&
        });
        curr_dim -= 1;
        dims -= 1;
    }
    quote! {
        #(#toks)* true
    }
}
