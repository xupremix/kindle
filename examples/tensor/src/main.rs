#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// use evol::candle::candle_nn::Module as _;
// use evol::candle::{candle_core, candle_nn};
use evol::prelude::*;

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let t: Tensor<Rank4<3, 4, 5, 6>> = Tensor::ones();
    let conv: Conv2d<4, 5> = Conv2d::conv2d(&vs, "hello", Default::default());
    let ris = conv.forward(&t);
    // let ris: Tensor<Rank4<3, 5, 3, 4>> = conv.forward(&t);
    println!("{}", ris);
}
