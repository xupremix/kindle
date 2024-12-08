use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

mod argmax;
mod argmin;
mod broadcast;
mod broadcast_as;
mod broadcast_left;
mod broadcast_matmul;
mod flatten;
mod flatten_from;
mod matmul;
mod squeeze_dim;
mod t;
mod transpose;
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
    let transpose = transpose::transpose(dims, name, idents, at_tk);
    let t = t::t(dims, name, idents, at_tk);
    let flatten = flatten::flatten(dims, name, idents, at_tk);
    let flatten_from = flatten_from::flatten_from(dims, name, idents, at_tk);
    let argmax = argmax::argmax(dims, name, idents, at_tk);
    let argmin = argmin::argmin(dims, name, idents, at_tk);
    let matmul = matmul::matmul(dims, name, idents, at_tk);
    let broadcast_matmul = broadcast_matmul::broadcast_matmul(dims, name, idents, at_tk);

    quote! {
        #broadcast
        #broadcast_as
        #broadcast_left
        #broadcast_matmul
        #squeeze_dim
        #unsqueeze
        #transpose
        #t
        #flatten
        #flatten_from
        #argmax
        #argmin
        #matmul
    }
}
