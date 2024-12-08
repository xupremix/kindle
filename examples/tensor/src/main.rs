use evol::{candle::candle_core, prelude::*};

fn main() {
    let t1: Tensor<Rank3<1, 3, 2>> = Tensor::ones();
    let t2: Tensor<Rank4<1, 3, 2, 3>> = Tensor::ones();
    let ris: Tensor<Rank4<1, 3, 3, 3>> = t1.broadcast_matmul(&t2);
}
