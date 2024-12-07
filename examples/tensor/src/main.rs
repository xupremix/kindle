// use evol::candle;
use evol::prelude::*;

fn main() {
    let t: Tensor<Rank2<2, 3>> = Tensor::ones();
    let t2: Tensor<Rank2<2, 3>> = Tensor::ones();
    let out = t + t2;
    println!("{}", out);
}
