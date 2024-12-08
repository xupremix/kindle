use evol::prelude::*;

fn main() {
    let t: Tensor<Rank6<2, 2, 3, 2, 3, 1>> = Tensor::ones();
    let t: Tensor<Rank6<2, 2, 1, 2, 3, 1>> = t.argmin_keepdim::<2>();
    println!("{}", t);
}
