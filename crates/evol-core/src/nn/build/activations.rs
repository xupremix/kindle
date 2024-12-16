use candle_nn::Module as _;

use super::ModelBuilder;
use crate::nn::layers::activations::*;

impl ModelBuilder for Gelu {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Gelu.forward(xs))
    }
}

impl ModelBuilder for NewGelu {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::NewGelu.forward(xs))
    }
}

impl ModelBuilder for Relu {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Relu.forward(xs))
    }
}

impl ModelBuilder for Relu2 {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Relu2.forward(xs))
    }
}

impl ModelBuilder for Relu6 {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Relu6.forward(xs))
    }
}

impl ModelBuilder for Silu {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Silu.forward(xs))
    }
}

impl ModelBuilder for Sigmoid {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Sigmoid.forward(xs))
    }
}

impl ModelBuilder for HardSigmoid {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::HardSigmoid.forward(xs))
    }
}

impl ModelBuilder for Swiglu {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Swiglu.forward(xs))
    }
}

impl ModelBuilder for Swish {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Swish.forward(xs))
    }
}

impl ModelBuilder for HardSwish {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(|xs| candle_nn::Activation::HardSwish.forward(xs))
    }
}

impl ModelBuilder for Elu {
    type Config = f64;
    fn step(
        _vs: &crate::prelude::Vs,
        cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::Elu(cfg).forward(xs))
    }
}

impl ModelBuilder for LeakyRelu {
    type Config = f64;
    fn step(
        _vs: &crate::prelude::Vs,
        cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(move |xs| candle_nn::Activation::LeakyRelu(cfg).forward(xs))
    }
}

impl ModelBuilder for GeluPytorchTanh {
    type Config = ();
    fn step(
        _vs: &crate::prelude::Vs,
        _cfg: Self::Config,
        seq: candle_nn::Sequential,
    ) -> candle_nn::Sequential {
        seq.add_fn(|xs| candle_nn::Activation::GeluPytorchTanh.forward(xs))
    }
}
