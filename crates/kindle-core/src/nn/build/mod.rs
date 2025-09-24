use crate::tensor::FromCandleTensor;
use std::marker::PhantomData;

use candle_nn::{seq, Module as _, Sequential};

use crate::tensor::ToCandleTensor;

use super::{vs::Vs, Module};

mod activations;
#[cfg(feature = "nightly")]
mod conv2d;
mod linear;
mod pool;
mod tuple;

pub struct Model<M> {
    repr: Sequential,
    module: PhantomData<M>,
}

pub trait ModelBuilder: Sized {
    type Config;
    fn build(vs: &Vs, cfg: Self::Config) -> Model<Self> {
        let repr = seq();
        let repr = Self::step(vs, cfg, repr);
        Model {
            repr,
            module: PhantomData,
        }
    }

    fn step(vs: &Vs, cfg: Self::Config, seq: Sequential) -> Sequential;
}

impl<T, M: Module<T>> Module<T> for Model<M>
where
    T: ToCandleTensor,
{
    type Output = M::Output;

    fn forward(&self, xs: &T) -> Self::Output {
        M::Output::from_candle_tensor(self.repr.forward(xs.to_candle_tensor()).unwrap())
    }
}
