#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use kindle::prelude::*;

type Model = (
    (Linear<10, 20>, Relu),
    (Linear<20, 30>, Relu),
    Linear<30, 2>,
);

#[derive(Module)]
struct Custom {
    l: Linear<10, 20>,
    r: Relu,
    l2: Linear<20, 30>,
    r2: Relu,
    l3: Linear<30, 2>,
}

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let model = Model::build(&vs, Default::default());
    let xs: Tensor<Rank3<2, 2, 10>> = Tensor::ones();

    // let xs = xs.flatten_all::<40>();
    println!("Before: \n{xs}");
    let xs = model.forward(&xs);
    println!("After: \n{xs}");
}
