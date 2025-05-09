#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use evol::prelude::*;

// model! {
//     Custom,
//     "/home/xupremix/Projects/evol/examples/tensor/src/model.onnx"
// }

dataset! {
    MnistTest,
    "/home/xupremix/Desktop/test.parquet"
}

#[derive(Module)]
struct CustomModel<K: Kind, D: Device> {
    first: Linear<784, 20, true, K, D>,
    second: Relu,
    third: Linear<20, 2, true, K, D>,
    _ignore_field: String,
    fourth: Swiglu,
    fifth: Linear<1, 2, true, K, D>,
}

fn main() {
    // testing byte conversion between candle and tch tensors for future model loading and tensor
    // switching
    // evol::testing();

    let dataset: MnistTest = MnistTest::load().unwrap();

    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let model = CustomModel {
        first: Linear::linear(&vs, "first"),
        second: Relu,
        third: Linear::linear(&vs, "third"),
        _ignore_field: "ignore".to_string(),
        fourth: Swiglu,
        fifth: Linear::linear(&vs, "fifth"),
    };
    let t: Tensor<Rank3<2, 2, 784>> = Tensor::random();
    let out = model.forward(&t);
    // println!("{}", out);
}
