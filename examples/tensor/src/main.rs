use evol::prelude::*;

type Custom = (Linear<4, 5>, Linear<5, 6>, Linear<6, 1>);

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let t: Tensor<Rank2<1, 4>> = Tensor::random();
    let model = Custom::build(&vs, Default::default());
    let mut sgd = Sgd::new(vm.all_vars(), 0.01);
    let loss = model.forward(&t);
    sgd.backward_step(&loss);

    println!("{}", loss);
}
