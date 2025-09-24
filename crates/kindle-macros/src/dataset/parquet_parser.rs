use std::path::Path;
use syn::Ident;

#[cfg(not(feature = "parquet"))]
pub(crate) fn parse_parquet(_: &Path) -> ([usize; 3], Ident, &'static str) {
    panic!("Parquet feature is not enabled");
}

#[cfg(feature = "parquet")]
pub(crate) fn parse_parquet(path: &Path) -> ([usize; 3], Ident, &'static str) {
    use image::GenericImageView;
    use parquet::{file::reader::FileReader as _, record::Field};
    use proc_macro2::Span;
    use std::fs::File;

    let file = File::open(path).unwrap();
    let reader = parquet::file::reader::SerializedFileReader::new(file).unwrap();
    let metadata = reader.metadata();
    let label_meta = metadata
        .row_group(0)
        .columns()
        .last()
        .unwrap()
        .statistics()
        .unwrap();
    let label_ty = match label_meta {
        parquet::file::statistics::Statistics::Boolean(_) => "bool",
        parquet::file::statistics::Statistics::Int32(_) => "i32",
        parquet::file::statistics::Statistics::Int64(_) => "i64",
        parquet::file::statistics::Statistics::Float(_) => "f32",
        parquet::file::statistics::Statistics::Double(_) => "f64",
        _ => panic!("unsupported label type"),
    };
    let file_metadata = metadata.file_metadata();
    let samples = file_metadata.num_rows() as usize; // first dimension

    let mut x = 0;
    let mut y = 0;
    let mut id = Ident::new("_", Span::call_site());
    let mut rows = reader.get_row_iter(None).unwrap();
    if let Some(Ok(row)) = rows.next() {
        for (col_name, field) in row.get_column_iter() {
            if let Field::Group(row) = field {
                for (_, data) in row.get_column_iter() {
                    if let Field::Bytes(bytes) = data {
                        if col_name == "image" {
                            // second and third dimension
                            let (xs, ys) =
                                image::load_from_memory(bytes.data()).unwrap().dimensions();
                            (x, y) = (xs as usize, ys as usize);
                            id = Ident::new("u8", Span::call_site());

                            // example of how to create a tensor from raw data, this could also be
                            // an option when switching from tch tensors to candle tensors
                            //
                            // let image = image::load_from_memory(bytes.data()).unwrap();
                            // let tensor = candle_core::Tensor::from_vec(
                            //     image.as_bytes().into(),
                            //     (x, y),
                            //     &candle_core::Device::Cpu,
                            // )
                            // .unwrap();
                            // println!("{:?}", tensor);
                        }
                    }
                }
            }
        }
    }

    if x == 0 || y == 0 {
        panic!("could find a valid image inside the parquet file")
    }

    ([samples, x, y], id, label_ty)
}
