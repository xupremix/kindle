use evol::prelude::*;

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let t: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let lin = Linear::<4, 5>::linear(&vs, "lin");
    let xs: Tensor<Rank3<2, 3, 5>> = lin.forward(&t);
    println!("{}", xs);
}
