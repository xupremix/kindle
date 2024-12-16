#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// use evol::candle::candle_nn::Module as _;
// use evol::candle::{candle_core, candle_nn};

use evol::prelude::*;

type Custom = (Linear<10, 20>, Relu, Conv2d<4, 5>, Relu, Linear<18, 30>);

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let xs: Tensor<Rank4<3, 4, 5, 10>> = Tensor::ones();
    let model = Custom::build(&vs, Default::default());
    let xs = model.forward(&xs);
    println!("{}", xs);
}
