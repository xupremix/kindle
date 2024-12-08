use evol::prelude::*;

fn main() {
    let v = [1, 2, 3, 4, 5, 6];
    let t2: Tensor<Rank2<2, 3>, u32> = Tensor::from_slice(&v);
    println!("{}", t2);
}
