use evol::prelude::*;

fn main() {
    let t2: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let t2: Tensor<Rank3<4, 3, 2>> = t2.transpose::<0, 2>();
    println!("{}", t2);
}
