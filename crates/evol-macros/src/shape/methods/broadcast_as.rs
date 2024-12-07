use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

// If the input shape is i_1, i_2, ... i_k,
// the target shape has to have k dimensions or more and shape j_1, ..., j_l, t_1, t_2, ..., t_k.
// The dimensions j_1 to j_l can have any value,
// the dimension t_a must be equal to i_a if i_a is different from 1.
// If i_a is equal to 1, any value can be used.

/*
impl<
    const D_D0: usize,
    const D_D1: usize,
    const D_D2: usize,
    const D_D3: usize,
    const D2: usize,
    const D3: usize
> BroadcastAs<
    Rank2<D2, D3>
> for Rank4<D_D0, D_D1, D_D2, D_D3>
{
    const BROADCAST_AS_CHECK: () = assert!(
        ( D_D2 == D2 || D2 == 1 ) &&
        ( D_D3 == D3 || D3 == 1 )
    );
}

impl<
    const D_D0: usize,
    const D_D1: usize,
    const D_D2: usize,
    const D_D3: usize,
    const D3: usize
> BroadcastAs<
    Rank1<D3>
> for Rank4<D_D0, D_D1, D_D2, D_D3>
{
    const BROADCAST_AS_CHECK: () = assert!(
        ( D_D3 == D3 || D3 == 1 )
    );
}
*/

pub(crate) fn broadcast_as(dims: usize, name: &TokenStream, at_tk: bool) -> TokenStream {
    if dims < 1 {
        return quote! {};
    }

    let mut toks = vec![];
    let mut curr_dim = 1;
    while curr_dim <= dims {
        let source_shape = Ident::new(&format!("Rank{}", curr_dim), Span::call_site());
        let target_shape = name;

        let target_const_dims = (0..dims)
            .map(|i| {
                let i = Ident::new(&format!("D_D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();

        let source_const_dims = (dims - curr_dim..dims)
            .map(|i| {
                let i = Ident::new(&format!("D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();

        let const_dims = quote! {
            #(#target_const_dims)*
            #(#source_const_dims)*
        };

        let target_idents = (0..dims)
            .map(|i| Ident::new(&format!("D_D{}", i), Span::call_site()))
            .collect::<Vec<_>>();

        let source_idents = (dims - curr_dim..dims)
            .map(|i| Ident::new(&format!("D{}", i), Span::call_site()))
            .collect::<Vec<_>>();

        let assert_msg = format!(
            "\nThe dimension provided for broadcasting {source_shape} into {target_shape} are not compatible.\nTo Broadcast when iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be:\n - equal\n - one of them is 1\n - one of them does not exist\n"
        );

        let assert_check = gen_assert_check(curr_dim, dims);

        let path = if at_tk {
            quote! {}
        } else {
            quote! { evol::prelude:: }
        };

        toks.push(quote! {
            impl<
                #const_dims
                > #path BroadcastAs<
                   #source_shape < #(#source_idents),* >
                > for #target_shape < #(#target_idents),* > {
                const BROADCAST_AS_CHECK: () = assert!(#assert_check, #assert_msg);
            }
        });
        curr_dim += 1;
    }
    quote! {
        #(#toks)*
    }
}

fn gen_assert_check(curr_dim: usize, dims: usize) -> TokenStream {
    // Reminder:
    // broadcast requirements:
    // When iterating over the dimension sizes, starting at the trailing dimension the dimension sizes must either be:
    // - equal,
    // - one of them is 1
    // - or one of them does not exist.
    let mut toks = vec![];
    for i in dims - curr_dim..dims {
        let source_dim = Ident::new(&format!("D{}", i), Span::call_site());
        let target_dim = Ident::new(&format!("D_D{}", i), Span::call_site());

        toks.push(quote! {
            (#source_dim == #target_dim || #source_dim == 1 ) &&
        })
    }
    quote! {
        #(#toks)* true
    }
}
