use super::ModelBuilder;
use crate::{nn::layers::pool::MaxPool2D, utils::ToUsize2};

impl<Pool: ToUsize2, Stride: ToUsize2> ModelBuilder for MaxPool2D<Pool, Stride> {
    type Config = ();

    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| xs.max_pool2d_with_stride(Pool::TUPLE, Stride::TUPLE))
    }
}
