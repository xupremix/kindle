use evol::prelude::*;

fn main() {
    let t1: Tensor<Rank2<2, 3>> = Tensor::ones();
    let t2: Tensor<Rank3<1, 2, 3>> = t1.zeros_like().reshape();
    println!("{}", t2.squeeze::<0>());
}
