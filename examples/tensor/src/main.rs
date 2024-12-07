use evol::prelude::*;

fn main() {
    let t1: Tensor<Rank2<2, 3>> = Tensor::ones();
    let t2: Tensor<Rank4<1, 2, 2, 3>> = t1.broadcast();
    println!("{}", t2);

    let t1: Tensor<Rank1<4>> = Tensor::ones();
    let t2 = t1.broadcast_left::<Rank2<2, 3>>();
    let s = t2.ones_like();
    let ris = s + t2 * 3 + 2;
    println!("{}", ris);
}
