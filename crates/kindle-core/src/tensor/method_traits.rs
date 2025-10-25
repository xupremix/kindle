use crate::shape::Shape;

// TODO: Remove impl for Scalar shape
pub trait DimInRange<const DIM: usize>: Shape {
    const DIM_IN_RANGE_CHECK: ();
}
impl<const DIM: usize, S: Shape> DimInRange<DIM> for S {
    const DIM_IN_RANGE_CHECK: () = assert!(DIM < S::DIMS, "Dimension index out of bounds");
}

pub trait Broadcast<Rhs: Shape, Dst: Shape>: Shape {
    const BROADCAST_CHECK: ();
}

pub trait BroadcastAs<Src: Shape>: Shape {
    const BROADCAST_AS_CHECK: ();
}

pub trait BroadcastLeft<Src: Shape>: Shape {
    type Extended: Shape;
}

pub trait Reshape<Src: Shape>: Shape {
    const RESHAPE_CHECK: () = assert!(
        Src::NELEMS == Self::NELEMS,
        "Reshape: number of elements must be the same"
    );
}
impl<S: Shape, D: Shape> Reshape<S> for D {}

pub trait SqueezeDim<const DIM: usize>: Shape {
    type SqueezeShape: Shape;
}

pub trait Unsqueeze<const DIM: usize>: Shape {
    type UnsqueezeShape: Shape;
}

pub trait T: Shape {
    type Transposed: Shape;
}

pub trait Transpose<const D0: usize, const D1: usize>: Shape {
    type Transposed: Shape;
}

pub trait FromSlice<T> {
    const FROM_SLICE_CHECK: ();
    fn from_slice(slice: T) -> Self;
}

pub trait FlattenAll<const N: usize>: Shape {
    const FLATTEN_ALL_CHECK: () = assert!(
        N == Self::NELEMS,
        "The flattened shape must have the same number of elements as the original shape"
    );
}
impl<const N: usize, S: Shape> FlattenAll<N> for S {}

pub trait Flatten<Src: Shape, const START: usize, const END: usize>: Shape {
    const FLATTEN_CHECK: ();
}

pub trait FlattenFrom<Src: Shape, const START: usize>: Shape {
    const FLATTEN_CHECK: ();
}

pub trait Argmax<const DIM: usize, const KEEP_DIM: bool>: Shape {
    type ArgmaxShape: Shape;
}

pub trait Argmin<const DIM: usize, const KEEP_DIM: bool>: Shape {
    type ArgminShape: Shape;
}

pub trait Matmul<Src: Shape>: Shape {
    type MatmulShape: Shape;
}

pub trait BroadcastMatmul<Src: Shape, Dst: Shape>: Shape {
    const BROADCAST_MATMUL_CHECK: ();
}

pub trait ToScalar: Shape {
    const TO_SCALAR_CHECK: () = assert!(Self::NELEMS == 1, "The tensor must have only 1 element");
}
impl<S: Shape> ToScalar for S {}

pub trait Indexer<S: Shape, const KEEP_DIM: bool> {
    type IndexShape: Shape;
    fn indexes() -> &'static [usize];
}

pub trait IndexOp<const DIM: usize, const KEEP_DIM: bool>: Shape {
    type IndexOpShape: Shape;
}

pub trait Get<const N: usize>: Shape {
    const GET_CHECK: ();
    type GetShape: Shape;
}

pub trait GetOnDim<const DIM: usize, const N: usize>: Shape {
    const GET_ON_DIM_CHECK: ();
    type GetOnDimShape: Shape;
}

pub trait Stack<const DIM: usize, const N: usize>: Shape {
    type StackShape: Shape;
}

pub trait Cat<const DIM: usize, const N: usize, Dst: Shape>: Shape {
    const CAT_CHECK: ();
}

pub trait Narrow<const DIM: usize, const START: usize, const LEN: usize>: Shape {
    const NARROW_CHECK: ();
    type NarrowShape: Shape;
}

pub trait Chunk<const DIM: usize, const NELEMS: usize>: Shape {
    const CHUNK_CHECK: ();
    type ChunkShape: Shape;
}

pub trait SwigluShape: Shape {
    type SwigluShape: Shape;
}

#[cfg(feature = "nightly")]
pub trait MaxPool2dShape: Shape {
    type MaxPool2dShape: Shape;
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use std::any::TypeId;

    macro_rules! gen_assert {
        ($name:ident, $code:block $($a:ty, $b:ty),* $(,)?) => {
            #[test]
            fn $name() {
                $code
                $(
                    assert_eq!(TypeId::of::<$a>(), TypeId::of::<$b>());
                )*
            }
        };
    }

    gen_assert! {
        matmul,
        {
            let t: Tensor<Rank2<1, 2>> = Tensor::ones();
            let t2: Tensor<Rank2<2, 3>> = Tensor::ones();
            let ris: Vec<f32> = t.matmul(&t2).data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let target = [[2.; 3]].concat();
            assert_eq!(ris, target);

        }
        <Rank2<2, 3> as Matmul<Rank2<1, 2>>>::MatmulShape,
        Rank2<1, 3>,
        <Rank3<1, 3, 4> as Matmul<Rank3<1, 2, 3>>>::MatmulShape,
        Rank3<1, 2, 4>,
        <Rank4<1, 2, 4, 5> as Matmul<Rank4<1, 2, 3, 4>>>::MatmulShape,
        Rank4<1, 2, 3, 5>,
        <Rank5<1, 2, 3, 5, 6> as Matmul<Rank5<1, 2, 3, 4, 5>>>::MatmulShape,
        Rank5<1, 2, 3, 4, 6>,
        <Rank6<1, 2, 3, 4, 6, 7> as Matmul<Rank6<1, 2, 3, 4, 5, 6>>>::MatmulShape,
        Rank6<1, 2, 3, 4, 5, 7>,
        <Rank7<1, 2, 3, 4, 5, 7, 8> as Matmul<Rank7<1, 2, 3, 4, 5, 6, 7>>>::MatmulShape,
        Rank7<1, 2, 3, 4, 5, 6, 8>,
        <Rank8<1, 2, 3, 4, 5, 6, 8, 9> as Matmul<Rank8<1, 2, 3, 4, 5, 6, 7, 8>>>::MatmulShape,
        Rank8<1, 2, 3, 4, 5, 6, 7, 9>,
    }
}
