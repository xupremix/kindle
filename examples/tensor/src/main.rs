// use evol::candle::candle_nn::{self};
use evol::prelude::*;

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let conv2d: Conv2d<10, 20> = Conv2d::conv2d(&vs, "hello", Default::default());
    println!("{:#?}", conv2d);
}
