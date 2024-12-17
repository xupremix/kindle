#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use evol::prelude::*;

type Custom = (Linear<784, 20>, Relu, Linear<20, 30>, Relu);

fn main() {
    let vm = VarMap::new();
    let vs: Vs = Vs::from_varmap(&vm);
    let dt = DataLoader::<Cuda>::mnist();
    let model = Custom::build(&vs, Default::default());
    let xs = model.forward(dt.train_images());
    println!("{:?}", xs);
}
