// use evol::candle::candle_core;
use evol::prelude::*;

fn main() {
    let t: Tensor<Rank3<3, 4, 5>> = Tensor::ones();
    let t2: Tensor<Rank3<3, 2, 5>> = t.narrow::<1, 1, 2>();
    println!("{}", t2);
}
