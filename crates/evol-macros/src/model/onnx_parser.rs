use std::path::Path;

#[cfg(not(feature = "onnx"))]
pub fn parse(_: &Path) {
    panic!("feature onnx is not enabled");
}

#[cfg(feature = "onnx")]
pub fn parse(path: &Path) {
    use candle_onnx as onnx;
    use std::collections::HashMap;

    println!("Parsing ONNX file: {:?}", path);
    let model = onnx::read_file(path).unwrap();
    let mut inputs = HashMap::new();
    let tensor = candle_core::Tensor::rand(0., 1., &[2, 3], &candle_core::Device::Cpu).unwrap();
    inputs.insert("input_ids".into(), tensor);
    let out = &onnx::simple_eval(&model, inputs);
    println!("Output: {:?}", out);
}
