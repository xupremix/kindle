use evol::prelude::*;

type Custom = (Linear<4, 5>, Linear<5, 6>, Linear<6, 2>);

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let t: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let model = Custom::build(&vs, Default::default());
    let xs = model.forward(&t);
    println!("{}", xs);
}
