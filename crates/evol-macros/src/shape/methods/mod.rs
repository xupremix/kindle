use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

mod broadcast;
mod broadcast_as;
mod broadcast_left;
mod squeeze_dim;
mod t;
mod unsqueeze;

pub(crate) fn methods(
    dims: usize,
    name: &TokenStream,
    idents: &[Ident],
    at_tk: bool,
) -> TokenStream {
    let broadcast = broadcast::broadcast(dims, name, at_tk);
    let broadcast_as = broadcast_as::broadcast_as(dims, name, at_tk);
    let broadcast_left = broadcast_left::broadcast_left(dims, name, at_tk);
    let squeeze_dim = squeeze_dim::squeeze_dim(dims, name, idents, at_tk);
    let unsqueeze = unsqueeze::unsqueeze(dims, name, idents, at_tk);
    let t = t::t(dims, name, idents, at_tk);

    quote! {
        #broadcast
        #broadcast_as
        #broadcast_left
        #squeeze_dim
        #unsqueeze
        #t
    }
}
