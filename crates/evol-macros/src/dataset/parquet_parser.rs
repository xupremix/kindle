#[cfg(not(feature = "parquet"))]
pub(crate) fn parse_parquet(_: &Path) {
    panic!("Parquet feature is not enabled");
}

#[cfg(feature = "parquet")]
use std::{fs::File, path::Path};

#[cfg(feature = "parquet")]
pub(crate) fn parse_parquet(path: &Path) -> [usize; 3] {
    use image::GenericImageView;
    use parquet::{file::reader::FileReader as _, record::Field};

    let file = File::open(path).unwrap();
    let reader = parquet::file::reader::SerializedFileReader::new(file).unwrap();
    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();
    let samples = file_metadata.num_rows() as usize; // first dimension

    let mut x = 0;
    let mut y = 0;
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

    [samples, x, y]
}
