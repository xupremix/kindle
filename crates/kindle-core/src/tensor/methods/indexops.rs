use crate::{
    prelude::IndexOp,
    tensor::{Device, Kind, Shape, Tensor},
};

macro_rules! single_dim_op {
    ($( [ $name:ident $kname:ident ] )*) => {
        $(
            impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
                pub fn $name<const DIM: usize>(&self) -> Tensor<S::IndexOpShape, K, D>
                where
                    S: IndexOp<DIM, false>,
                {
                    Tensor {
                        repr: self.repr.$name(DIM).unwrap(),
                        ..Default::default()
                    }
                }

                pub fn $kname<const DIM: usize>(&self) -> Tensor<S::IndexOpShape, K, D>
                where
                    S: IndexOp<DIM, true>,
                {
                    Tensor {
                        repr: self.repr.$kname(DIM).unwrap(),
                        ..Default::default()
                    }
                }
            }
        )*
    };
    (u32 $( [ $name:ident $kname:ident ] )*) => {
        $(
            impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
                pub fn $name<const DIM: usize>(&self) -> Tensor<S::IndexOpShape, u32, D>
                where
                    S: IndexOp<DIM, false>,
                {
                    Tensor {
                        repr: self.repr.$name(DIM).unwrap(),
                        ..Default::default()
                    }
                }

                pub fn $kname<const DIM: usize>(&self) -> Tensor<S::IndexOpShape, u32, D>
                where
                    S: IndexOp<DIM, true>,
                {
                    Tensor {
                        repr: self.repr.$kname(DIM).unwrap(),
                        ..Default::default()
                    }
                }
            }
        )*
    }
}

single_dim_op! {
    [ max max_keepdim ]
    [ min min_keepdim ]
    [ var var_keepdim ]
}
single_dim_op! {
    u32
    [ argmin argmin_keepdim ]
    [ argmax argmax_keepdim ]
}
