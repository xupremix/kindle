// #![allow(incomplete_features)]
// #![feature(generic_const_exprs)]

use evol::prelude::*;

type Custom = (Linear<784, 20>, Relu, Linear<20, 2>, Swiglu);

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    println!("Loading MNIST dataset...");
    let dt = DataLoader::<Cuda>::mnist();
    println!("Building model...");
    let model = Custom::build(&vs, Default::default());
    println!("Forwarding model...");
    let xs = model.forward(dt.train_images());
    println!("Model output:");
    println!("{:?}", xs);
}
