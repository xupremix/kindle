// use evol::candle::candle_core;
use evol::prelude::*;

fn main() {
    let t1: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let t2: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let t3: Tensor<Rank3<2, 3, 4>> = Tensor::ones();

    let ris = Tensor::stack::<3, 3>(&[&t1, &t2, &t3]);
    println!("{}", ris);
}
