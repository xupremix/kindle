#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

type Custom = (Linear<10, 32>, Conv2d<20, 30>, Linear<30, 40>);

// use evol::candle::candle_nn::Module as _;
// use evol::candle::{candle_core, candle_nn};
use evol::prelude::*;

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let t: Tensor<Rank4<3, 20, 5, 10>> = Tensor::ones();
    let model = Custom::build(&vs, Default::default());
    let xs = model.forward(&t);
    println!("{}", xs);
}
