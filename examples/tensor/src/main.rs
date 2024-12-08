use evol::prelude::*;

fn main() {
    let t2: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let t2 = t2.t();
    println!("{}", t2);
}
