use proc_macro::TokenStream;

mod model;
mod shape;

#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream {
    shape::shape(input)
}

// Model
#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    model::model(input)
}

// Dataset
// Module
