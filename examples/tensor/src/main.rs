// use evol::candle::candle_core;
use evol::prelude::*;

fn main() {
    let t: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let ris: Tensor<Rank2<3, 4>> = t.get::<2>();
    println!("{}", ris);
}
