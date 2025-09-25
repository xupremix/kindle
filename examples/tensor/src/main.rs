#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use kindle::prelude::*;

dataset! {
    MnistDataset,
    "/home/xupremix/Desktop/train.parquet"
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
    let t: Tensor<Rank4<2, 3, 4, 5>> = Tensor::random();
    let maxpool = t
        .to_candle_tensor()
        .max_pool2d_with_stride((3, 3), (1, 1))
        .unwrap();
    let out: Tensor<Rank4<2, 3, 2, 3>> = MaxPool2D::default().forward(&t);
    println!("{}", out);
    println!("{}", maxpool);

    let dataset: MnistDataset = MnistDataset::load().unwrap();
    println!("{:?}", dataset.images);

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
    let _out = model.forward(&t);
    // println!("{}", out);
}

// model! {
//     Custom,
//     "/home/xupremix/Projects/evol/examples/tensor/src/model.onnx"
// }
