#[cfg(feature = "onnx")]
pub(crate) fn parse_file(path: String) {
    let _ = path;
    // println!("ONNX");
}

#[cfg(not(feature = "onnx"))]
pub(crate) fn parse_file(path: String) {
    let _ = path;
    // println!("NOT ONNX");
}
