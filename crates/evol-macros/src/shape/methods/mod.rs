use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

mod broadcast;

pub(crate) fn methods(
    dims: usize,
    name: &TokenStream,
    _idents: &[Ident],
    at_tk: bool,
) -> TokenStream {
    let broadcast = broadcast::broadcast(dims, name, at_tk);
    quote! {
        #broadcast
    }
}
