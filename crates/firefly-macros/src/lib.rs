use proc_macro::TokenStream;

mod shape;

#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream {
    shape::shape(input)
}

// Model
// Dataset
// Module
