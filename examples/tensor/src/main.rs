use evol::prelude::*;

fn main() {
    let t1: Tensor<Rank2<2, 3>> = Tensor::ones();
    let t2: Tensor<Rank3<1, 2, 3>> = t1.zeros_like().reshape();
    let ris = t2.reshape_like(&t1);
    println!("{}", ris);
}
