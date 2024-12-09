use candle_core::{Var, WithDType};
use candle_nn::Optimizer;

use crate::{device::Device, shape::Shape, tensor::Tensor};

pub trait Backward<T> {
    const BACKWARD_CHECK: ();
    fn backward_step(&mut self, loss: &T);
}

pub struct Sgd {
    repr: candle_nn::SGD,
}

impl Sgd {
    pub fn new(vars: Vec<Var>, lr: f64) -> Self {
        Self {
            repr: candle_nn::SGD::new(vars, lr).unwrap(),
        }
    }

    pub fn lr(&self) -> f64 {
        self.repr.learning_rate()
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.repr.set_learning_rate(lr)
    }

    // TODO: look into the GradStore type and how to implement it
}

impl<S: Shape, K: WithDType, D: Device> Backward<Tensor<S, K, D>> for Sgd {
    const BACKWARD_CHECK: () = assert!(
        S::NELEMS == 1,
        "The loss must be a Scalar or a Tensor with only 1 element"
    );

    fn backward_step(&mut self, loss: &Tensor<S, K, D>) {
        <Sgd as Backward<Tensor<S, K, D>>>::BACKWARD_CHECK;
        self.repr.backward_step(&loss.repr).unwrap();
    }
}
