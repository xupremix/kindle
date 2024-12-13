use evol::prelude::*;

fn main() {
    let t: Tensor<Rank4<2, 3, 4, 5>> = Tensor::ones();
    let ris = t.argmin::<0>();
    println!("{}", ris);
}
