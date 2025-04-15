#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use evol::prelude::*;

model! {
    Custom,
    "/home/xupremix/Projects/evol/examples/tensor/src/model.safetensors"
}

// type MnistModel = (Linear<784, 20>, Relu, Linear<20, 2>, Swiglu);
// type CifarModel = (Linear<32, 20>, Relu, Linear<20, 2>, Swiglu);

#[derive(Module)]
struct CustomModel<K: Kind> {
    first: Linear<784, 20, true, K, Cuda<0>>,
    second: Relu,
    third: Linear<20, 2, true, K, Cuda<0>>,
    _another: String,
    fourth: Swiglu,
    fifth: Linear<1, 2, true, K, Cuda<0>>,
}

/*
impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for CustomModel<D, K>
where
    S: Forward<784, 20>,
    <S as Forward<784, 20>>::ForwardShape: Forward<20, 2>,
    <<S as Forward<784, 20>>::ForwardShape as Forward<20, 2>>::ForwardShape: SwigluShape,
    <<<S as Forward<784, 20>>::ForwardShape as Forward<20, 2>>::ForwardShape as SwigluShape>::SwigluShape: Forward<1, 2>
{
    type Output = Tensor<<<<<S as Forward<784, 20>>::ForwardShape as Forward<20, 2>>::ForwardShape as SwigluShape>::SwigluShape as Forward<1, 2>>::ForwardShape, K, D>;

    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        let xs = self.first.forward(xs);
        let xs = self.second.forward(&xs);
        let xs = self.third.forward(&xs);
        let xs = self.fourth.forward(&xs);
        self.fifth.forward(&xs)
    }
}
*/

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let model = CustomModel {
        first: Linear::linear(&vs, "first"),
        second: Relu,
        third: Linear::linear(&vs, "third"),
        _another: "another".to_string(),
        fourth: Swiglu,
        fifth: Linear::linear(&vs, "fifth"),
    };
    let t: Tensor<Rank3<2, 2, 784>, _, _> = Tensor::random();
    let out: Tensor<Rank3<2, 2, 2>, _, _> = model.forward(&t);
    println!("{}", out);
}
