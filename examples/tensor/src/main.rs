use evol::prelude::*;

fn main() {
    let t: Tensor<Rank6<2, 2, 3, 2, 3, 1>> = Tensor::ones();
    let t = t.flatten_from::<3, Rank4<2, 2, 3, 6>>();
    println!("{}", t);
}
