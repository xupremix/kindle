// use evol::candle::candle_core;
use evol::prelude::*;

fn main() {
    let t1: Tensor<Rank2<2, 3>> = Tensor::new([[1., 2., 3.], [4., 5., 6.]]);
    let t2: Tensor<Rank2<2, 3>> = Tensor::new([[1., 3., 3.], [4., 5., 6.]]);
    let out = t1 == t2;
    println!("{}", out);
}
