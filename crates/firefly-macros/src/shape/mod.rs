use arg::Args;
use proc_macro::TokenStream;
use quote::quote;

mod arg;

pub(crate) fn shape(input: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(input as Args);
    println!("{:?}", args);
    quote! {}.into()
}
