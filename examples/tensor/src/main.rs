#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use evol::prelude::*;

// type MnistModel = (Linear<784, 20>, Relu, Linear<20, 2>, Swiglu);
type CifarModel = (Linear<32, 20>, Relu, Linear<20, 2>, Swiglu);

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    println!("Loading CIFAR dataset...");
    let dt = DataLoader::<Cpu>::cifar();
    println!("Building model...");
    let model = CifarModel::build(&vs, Default::default());
    println!("Forwarding model...");
    let xs = model.forward(&dt.train_images().to_kind::<f32>());
    println!("Model output:");
    println!("{:?}", xs);
}
