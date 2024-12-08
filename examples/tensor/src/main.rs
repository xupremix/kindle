use evol::prelude::*;

fn main() {
    let t1: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let t2: Tensor<Rank3<2, 4, 5>> = Tensor::ones();
    let t3 = t1.matmul(&t2);
    println!("{}", t3);
}
