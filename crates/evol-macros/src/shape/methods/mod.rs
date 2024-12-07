use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

mod broadcast;
mod broadcast_as;
mod broadcast_left;

pub(crate) fn methods(
    dims: usize,
    name: &TokenStream,
    _idents: &[Ident],
    at_tk: bool,
) -> TokenStream {
    let broadcast = broadcast::broadcast(dims, name, at_tk);
    let broadcast_as = broadcast_as::broadcast_as(dims, name, at_tk);
    let broadcast_left = broadcast_left::broadcast_left(dims, name, at_tk);
    quote! {
        #broadcast
        #broadcast_as
        #broadcast_left
    }
}
