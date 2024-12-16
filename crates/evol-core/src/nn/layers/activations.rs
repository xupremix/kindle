use crate::prelude::{Device, Kind, Module, Shape, Tensor};
use candle_nn::Module as _;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gelu;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Gelu {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Gelu.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewGelu;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for NewGelu {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::NewGelu.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Relu;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Relu {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Relu.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Relu2;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Relu2 {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Relu2.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Relu6;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Relu6 {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Relu6.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Silu;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Silu {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Silu.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sigmoid;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Sigmoid {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Sigmoid.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardSigmoid;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for HardSigmoid {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::HardSigmoid
                .forward(&xs.repr)
                .unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Swiglu;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Swiglu {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Swiglu.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Swish;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Swish {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Swish.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HardSwish;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for HardSwish {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::HardSwish.forward(&xs.repr).unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Elu(f64);

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for Elu {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::Elu(self.0)
                .forward(&xs.repr)
                .unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeakyRelu(f64);

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for LeakyRelu {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::LeakyRelu(self.0)
                .forward(&xs.repr)
                .unwrap(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeluPytorchTanh;

impl<S: Shape, K: Kind, D: Device> Module<Tensor<S, K, D>> for GeluPytorchTanh {
    type Output = Tensor<S, K, D>;
    fn forward(&self, xs: &Tensor<S, K, D>) -> Self::Output {
        Tensor {
            repr: candle_nn::Activation::GeluPytorchTanh
                .forward(&xs.repr)
                .unwrap(),
            ..Default::default()
        }
    }
}
