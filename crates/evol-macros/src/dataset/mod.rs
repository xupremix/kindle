use proc_macro::TokenStream;
use quote::quote;
use std::path::Path;
use syn::Ident;

mod arg;
mod parquet_parser;
use arg::Args;

pub(crate) fn dataset(input: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(input as Args);
    let name = args.name;
    let path_str = &args.path;
    let path = Path::new(&args.path);
    let (dims, kind) = parse(path);
    let elems: usize = dims.iter().product();

    #[cfg(feature = "cuda")]
    let device = quote! { Cuda };

    #[cfg(not(feature = "cuda"))]
    let device = quote! { Cpu };

    quote! {
        #[derive(Debug, Clone)]
        struct #name <
            D: evol::device::Device = evol::device:: #device ,
        > {
            images: evol::tensor::Tensor<
                evol::shape::Rank3<#(#dims),*> ,
                #kind,
                D,
            >,
        }

        impl<
            D: evol::device::Device,
        > #name <D> {
            pub fn load() -> std::io::Result<Self> {
                use evol::image::GenericImageView;
                use evol::parquet::{file::reader::FileReader as _, };
                use evol::parquet::data_type::AsBytes;

                let mut buffer_data: Vec<#kind> = Vec::with_capacity(#elems);
                // let mut buffer_data_labels: Vec<#kind> = Vec::with_capacity();

                let file = std::fs::File::open(#path_str)?;
                let reader = evol::parquet::file::reader::SerializedFileReader::new(file)?;
                let rows = reader.get_row_iter(None)?;

                for row in rows {
                    for (col_name, field) in row?.get_column_iter() {
                        if let evol::parquet::record::Field::Group(srow) = field {
                            for (_, field) in srow.get_column_iter() {
                                if let evol::parquet::record::Field::Bytes(data) = field {
                                    let image = evol::image::load_from_memory(data.data()).unwrap();
                                    buffer_data.extend(image.to_luma8().as_bytes());
                                }
                            }
                        } else if let evol::parquet::record::Field::Long(label) = field {
                            // parsing a label
                        }

                    }
                }

                let ptr: Box<[#kind; #elems]> = buffer_data.into_boxed_slice().try_into().unwrap();
                let images = evol::tensor::Tensor::from_slice(&*ptr);

                std::io::Result::Ok(
                    Self {
                        images,
                    }
                )
            }
        }
    }
    .into()
}

fn parse(path: &Path) -> ([usize; 3], Ident) {
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
