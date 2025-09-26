#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use kindle::prelude::*;

dataset! {
    MnistDataset,
    "/home/xupremix/Desktop/train.parquet"
}

// #[derive(Module)]
// struct MnistModel {
//     conv1: Conv2d<1, 32>,
//     relu1: Relu,
//     maxp1: MaxPool2D<Kernel<2>, Stride<0, 2>>,
//     conv2: Conv2d<32, 64>,
//     relu2: Relu,
//     maxp2: MaxPool2D<Kernel<2>, Stride<0, 2>>,
//     // flatten
//     // dropout
//     fc: Linear<1600, 10>,
//     softmax: SoftmaxLastDim,
// }

// type MyModel = (
// Conv2d<1, 32>,
// Relu,
// MaxPool2D<Kernel<2>, Stride<1>>,
// Conv2d<32, 64>,
// Relu,
// MaxPool2D<Kernel<2>, Stride<1>>,
// );

fn main() {
    let t = Tensor::<Rank1<10>>::dyn_one_hot(9).unwrap();
    println!("{t}");

    let dataset: MnistDataset = MnistDataset::load().unwrap();
    let tensors = dataset.images.unsqueeze::<1>().to_kind::<f32>();
    let labels = dataset.labels.chunk::<0, 1>();
    let label = labels[0].squeeze::<0>().to_scalar();
    let xs = &tensors.chunk::<0, 1>()[0];

    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);

    let conv1 = Conv2d::<1, 32>::conv2d(&vs, "conv1");
    let maxp1 = MaxPool2D::<Kernel<2>>::new();
    let conv2 = Conv2d::<32, 64>::conv2d(&vs, "conv2");
    let maxp2 = MaxPool2D::<Kernel<2>>::new();
    let fc = Linear::<30976, 10>::linear(&vs, "fc");

    let xs = conv1.forward(xs).relu();
    let xs = maxp1.forward(&xs);
    let xs = conv2.forward(&xs).relu();
    let xs = maxp2
        .forward(&xs)
        .flatten_all()
        .dropout(0.5)
        .unsqueeze::<0>();
    let xs = fc.forward(&xs).softmax::<1>();

    println!("{}", xs);
}

// model! {
//     Custom,
//     "/home/xupremix/Projects/evol/examples/tensor/src/model.onnx"
// }
