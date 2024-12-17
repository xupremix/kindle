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
