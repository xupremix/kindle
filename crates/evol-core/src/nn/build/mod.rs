use crate::tensor::FromCandleTensor;
use std::marker::PhantomData;

use candle_nn::{seq, Module as _, Sequential};

use crate::tensor::ToCandleTensor;

use super::{vs::Vs, Module};

#[cfg(feature = "nightly")]
mod conv2d;
mod linear;

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

impl<M0: ModelBuilder, M1: ModelBuilder, M2: ModelBuilder> ModelBuilder for (M0, M1, M2) {
    type Config = (M0::Config, M1::Config, M2::Config);

    fn step(vs: &Vs, c: Self::Config, seq: Sequential) -> Sequential {
        let (c0, c1, c2) = c;
        let seq = M0::step(vs, c0, seq);
        let seq = M1::step(vs, c1, seq);
        M2::step(vs, c2, seq)
    }
}
