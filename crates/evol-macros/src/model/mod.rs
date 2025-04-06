use proc_macro::TokenStream;
use quote::quote;

mod arg;
mod parser;
use arg::Args;

pub(crate) fn model(input: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(input as Args);
    let name = args.name;
    let path = args.path;

    parser::parse_file(path);

    quote! {
        #[derive(Debug, Clone)]
        struct #name {

        }

        impl #name {
            pub fn example() {
                println!("Hello World");
            }
        }
    }
    .into()
}
