use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

/*
impl<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
    const ND0: usize,
    const TD0: usize,
> BroadcastMatmul<Rank5<D0, D1, D2, D3, D4>, Rank5<D0, D1, TD0, D3, D5>> for Rank3<ND0, D4, D5> {
    const BROADCAST_MATMUL_CHECK: () = assert!(
        (TD0 == D2 || TD0 == ND0) &&
        (TD0 >= D2 && TD0 >= ND0) &&
        (ND0 == D2 || ND0 == 1 || D2 == 1),
        "Broadcast Matmul check failed for dimensions 2 to 5"
    );
}
*/

fn gen_assert_check(dims: usize, curr: usize) -> TokenStream {
    let start = dims - curr;
    let end = dims - 2;

    let mut out = vec![];

    (start..end).enumerate().for_each(|(i, k)| {
        let d = Ident::new(&format!("D{}", k), Span::call_site());
        let n = Ident::new(&format!("ND{}", i), Span::call_site());
        let t = Ident::new(&format!("TD{}", i), Span::call_site());

        out.push(quote! {
            (#t == #d || #t == #n) &&
            (#t >= #d && #t >= #n) &&
            (#n == #d || #n == 1 || #d == 1) &&
        });
    });

    quote! { #(#out)* true }
}

pub(crate) fn broadcast_matmul(
    dims: usize,
    name: &TokenStream,
    idents: &[Ident],
    at_tk: bool,
) -> TokenStream {
    if dims < 2 {
        return quote! {};
    }

    let path = if at_tk {
        quote! {}
    } else {
        quote! { evol::prelude:: }
    };

    let mut toks = vec![];
    let mut curr_dim = 2;

    while curr_dim <= dims {
        let source_shape = name;
        let mut source_idents = idents.to_vec();

        let shape_rhs = Ident::new(&format!("Rank{}", curr_dim), Span::call_site());
        let mut rhs_idents = (0..curr_dim - 2)
            .map(|i| Ident::new(&format!("ND{}", i), Span::call_site()))
            .collect::<Vec<_>>();
        rhs_idents.extend_from_slice(&[
            Ident::new(&format!("D{}", dims - 1), Span::call_site()),
            Ident::new(&format!("D{}", dims), Span::call_site()),
        ]);

        let target_shape = name;
        let mut target_idents = if curr_dim == dims {
            vec![]
        } else {
            (0..dims - curr_dim)
                .map(|i| Ident::new(&format!("D{}", i), Span::call_site()))
                .collect::<Vec<_>>()
        };
        target_idents
            .extend((0..curr_dim - 2).map(|i| Ident::new(&format!("TD{}", i), Span::call_site())));
        target_idents.extend_from_slice(&[
            Ident::new(&format!("D{}", dims - 2), Span::call_site()),
            Ident::new(&format!("D{}", dims), Span::call_site()),
        ]);

        let mut source_const_dims = idents
            .iter()
            .map(|i| {
                quote! {
                    const #i: usize
                }
            })
            .collect::<Vec<_>>();
        source_const_dims.extend({
            let i = Ident::new(&format!("D{}", dims), Span::call_site());
            [quote! {
                const #i: usize
            }]
        });

        let const_new_dims = (0..curr_dim - 2)
            .map(|i| {
                let i = Ident::new(&format!("ND{}", i), Span::call_site());
                quote! {
                    const #i: usize
                }
            })
            .collect::<Vec<_>>();

        let const_target_dims = (0..curr_dim - 2)
            .map(|i| {
                let i = Ident::new(&format!("TD{}", i), Span::call_site());
                quote! {
                    const #i: usize
                }
            })
            .collect::<Vec<_>>();

        let const_dims = quote! {
            #(#source_const_dims,)*
            #(#const_new_dims,)*
            #(#const_target_dims),*
        };

        let assert_msg =
            format!("Broadcast Matmul check failed for dimensions {curr_dim} to {dims}");

        let assert_check = gen_assert_check(dims, curr_dim);

        toks.push(quote! {
            impl<
                #const_dims
            > #path BroadcastMatmul<
                #source_shape<#(#source_idents),*>,
                #target_shape<#(#target_idents),*>
            > for #shape_rhs<#(#rhs_idents),*> {
                const BROADCAST_MATMUL_CHECK: () = assert!(#assert_check, #assert_msg);
            }
        });

        if curr_dim != dims {
            target_idents[dims - 2] = source_idents[dims - 1].clone();
            let last = target_idents[dims - 1].clone();
            target_idents[dims - 1] = source_idents[dims - 2].clone();
            source_idents[dims - 1] = last;
            source_idents.swap(dims - 1, dims - 2);

            toks.push(quote! {
                impl<
                    #const_dims
                > #path BroadcastMatmul<
                    #shape_rhs<#(#rhs_idents),*>,
                    #target_shape<#(#target_idents),*>
                > for #source_shape<#(#source_idents),*> {
                    const BROADCAST_MATMUL_CHECK: () = assert!(#assert_check, #assert_msg);
                }
            });
        }

        curr_dim += 1;
    }

    quote! {
        #(#toks)*
    }
}
