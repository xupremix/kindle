use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

// mod i;
mod indexer;

pub(crate) fn index(dims: usize, idents: &[Ident], at_tk: bool) -> TokenStream {
    let indexer = indexer::indexer(dims, idents, at_tk);
    // let i = i::i(dims, idents, at_tk);
    quote! {
        #indexer
        // #i
    }
}
