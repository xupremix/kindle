#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use kindle::prelude::*;

fn main() {
    #[derive(Module)]
    struct Model {
        l: Linear<10, 20>,
        r: Relu,
        l2: Linear<20, 2>,
    }
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let model = Model {
        l: Linear::linear(&vs, "l"),
        r: Relu,
        l2: Linear::linear(&vs, "l2"),
    };
    let xs: Tensor<Rank2<2, 10>> = Tensor::ones();
    let xs = model.forward(&xs);
}
