use evol::prelude::*;

fn main() {
    let t1: Tensor<Rank2<1, 1>> = Tensor::ones();
    let t2: Tensor<Rank2<2, 3>> = Tensor::ones();

    let ris: Tensor<Rank2<2, 3>, _> = t1.broadcast_eq(&t2);
    println!("{}", ris);
}
