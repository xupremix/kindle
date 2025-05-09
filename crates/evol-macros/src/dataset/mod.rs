use std::path::Path;

use proc_macro::TokenStream;
use quote::quote;

mod arg;
mod parquet_parser;
use arg::Args;

pub(crate) fn dataset(input: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(input as Args);
    let name = args.name;
    let path = Path::new(&args.path);
    let dims = parse(path);

    #[cfg(feature = "cuda")]
    let device = quote! { Cuda };

    #[cfg(not(feature = "cuda"))]
    let device = quote! { Cpu };

    quote! {
        #[derive(Debug, Clone)]
        struct #name <
            K: evol::kind::Kind = f32,
            D: evol::device::Device = evol::device:: #device ,
        > {
            data: evol::tensor::Tensor<
                evol::shape::Rank3<#(#dims),*> ,
                K,
                D,
            >,
        }

        impl<
            K: evol::kind::Kind,
            D: evol::device::Device,
        > #name <K, D> {
            pub fn load() -> Self {
                todo!()
            }
        }
    }
    .into()
}

fn parse(path: &Path) -> [usize; 3] {
    match path
        .extension()
        .expect("File extension not found")
        .to_str()
        .expect("Ivalid string")
    {
        "parquet" => parquet_parser::parse_parquet(path),
        _ => panic!("Unsupported file format"),
    }
}
