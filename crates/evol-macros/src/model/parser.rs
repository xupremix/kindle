use std::{collections::HashMap, path::Path};

#[cfg(feature = "onnx")]
use candle_onnx as onnx;

#[cfg(feature = "onnx")]
pub(crate) fn parse_file(path: String) {
    parse(path, true);
}

#[cfg(not(feature = "onnx"))]
pub(crate) fn parse_file(path: String) {
    parse(path, false);
}

fn parse(path: String, onnx_enabled: bool) {
    let path = Path::new(&path);
    match path
        .extension()
        .expect("File extension not found")
        .to_str()
        .expect("Ivalid string")
    {
        "onnx" if onnx_enabled => parse_onnx(path),
        "onnx" if !onnx_enabled => panic!("onnx feature is not enabled"),
        "pth" | "pt" => parse_pth(path),
        "pb" => todo!(),
        "keras" => todo!(),
        "h5" | "hdf5" => todo!(),
        "pkl" | "pickle" => todo!(),
        "npy" | "npz" => todo!(),
        _ => panic!("Unsupported file format"),
    }
}

fn parse_onnx(path: &Path) {
    println!("Parsing ONNX file: {:?}", path);
    let model = onnx::read_file(path).unwrap();
    let mut inputs = HashMap::new();
    let tensor = candle_core::Tensor::rand(0., 1., &[2, 3], &candle_core::Device::Cpu).unwrap();
    inputs.insert("input_ids".into(), tensor);
    let out = &onnx::simple_eval(&model, inputs);
    println!("Output: {:?}", out);
}

fn parse_pth(path: &Path) {}
// fn parse_tf(path: &Path) {}
// fn parse_keras(path: &Path) {}
// fn parse_hdf5(path: &Path) {}
// fn parse_pickle(path: &Path) {}
// fn parse_numpy(path: &Path) {}
