use crate::tensor::{Device, Kind, Shape, Tensor};

macro_rules! wrap {
    ($( $f:ident )*) => {
        impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
            $(
                pub fn $f(&self) -> Self {
                    Self {
                        repr: self.repr.$f().unwrap(),
                        ..Default::default()
                    }
                }
            )*
        }
    };
}

// TODO: CHECK IMPL WHEN DTYPE != FLOAT

wrap! {
    neg
    abs
    exp
    cos
    erf
    floor
    gelu
    gelu_erf
    log
    relu
    round
    silu
    sin
    sqr
    sqrt
    tanh
    ceil
}
