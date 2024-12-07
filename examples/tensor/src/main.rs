// use evol::candle::candle_core;
use evol::{candle::candle_core, prelude::*};

fn main() {
    let t1: Tensor<Rank2<2, 1>> = Tensor::ones();
    let t2: Tensor<Rank2<1, 3>> = Tensor::ones();
    let ris: Tensor<Rank2<2, 3>> = t1.broadcast_add(&t2);
    println!("{}", ris);

    // let t1: Tensor<Rank2<2, 1>> = Tensor::ones();
    // let t2: Tensor<Rank2<2, 3>> = t1.broadcast();
}
