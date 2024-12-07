use evol::prelude::*;

fn main() {
    let t: Tensor<Rank2<2, 3>> = Tensor::from_slice(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    println!("{:#?}", t.to_vec());
    let t: Tensor<Rank2<2, 3>> = Tensor::new([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]]);
    println!("{:#?}", t.to_vec());
}
