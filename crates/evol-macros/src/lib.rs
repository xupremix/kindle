use proc_macro::TokenStream;

mod dataset;
mod model;
mod module;
mod shape;

#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream {
    shape::shape(input)
}

// Module
#[proc_macro_derive(Module)]
pub fn module(input: TokenStream) -> TokenStream {
    module::module(input)
}

// Model
#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    model::model(input)
}

// Dataset
#[proc_macro]
pub fn dataset(input: TokenStream) -> TokenStream {
    dataset::dataset(input)
}
