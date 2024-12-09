use crate::{
    prelude::ToScalar,
    shape::Scalar,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: ToScalar, K: Kind, D: Device> Tensor<Scalar, K, D> {
    pub fn to_scalar(&self) -> K {
        S::TO_SCALAR_CHECK;
        self.repr.to_scalar().unwrap()
    }
}
