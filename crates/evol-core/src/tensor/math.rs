use crate::tensor::{Device, Kind, Shape, Tensor};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    pub fn elu(&self, alpha: f64) -> Self {
        Self {
            repr: self.repr.elu(alpha).unwrap(),
            ..Default::default()
        }
    }

    pub fn powf(&self, exp: f64) -> Self {
        Self {
            repr: self.repr.powf(exp).unwrap(),
            ..Default::default()
        }
    }

    pub fn affine(&self, mul: f64, add: f64) -> Self {
        Self {
            repr: self.repr.affine(mul, add).unwrap(),
            ..Default::default()
        }
    }
}
