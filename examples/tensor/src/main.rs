// use evol::candle::candle_core;
use evol::prelude::*;

fn main() {
    let t: Tensor<Rank3<3, 4, 6>> = Tensor::ones();
    let ris: Vec<Tensor<Rank3<3, 4, 2>>> = t.chunk::<2, 2>();
    println!("{:#?}", ris);
}
