#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// use evol::candle::candle_nn::Module as _;
// use evol::candle::{candle_core, candle_nn};
use evol::prelude::*;

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let xs: Tensor<Rank4<3, 4, 5, 10>> = Tensor::ones();
    let lin: Linear<10, 20> = Linear::linear(&vs, "hello");
    let xs = lin.forward(&xs);
    println!("{}", xs);
    let conv: Conv2d<4, 5> = Conv2d::conv2d(&vs, "hello");
    let xs = conv.forward(&xs);
    // let xs: Tensor<Rank4<3, 5, 3, 18>> = conv.forward(&xs);
    println!("{}", xs);
    // let ris: Tensor<Rank4<3, 5, 3, 4>> = conv.forward(&t);
}
