use proc_macro::TokenStream;
use quote::quote;
use std::path::Path;

mod arg;
mod onnx_parser;

use arg::Args;

pub(crate) fn model(input: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(input as Args);
    let name = args.name;
    let path = args.path;

    parse(path);

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

fn parse(path: String) {
    let path = Path::new(&path);
    match path
        .extension()
        .expect("File extension not found")
        .to_str()
        .expect("Ivalid string")
    {
        "onnx" => onnx_parser::parse(path),
        "pth" | "pt" => todo!(),
        "pb" => todo!(),
        "keras" => todo!(),
        "h5" | "hdf5" => todo!(),
        "pkl" | "pickle" => todo!(),
        "npy" | "npz" => todo!(),
        _ => panic!("Unsupported file format"),
    }
}
