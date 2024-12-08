use crate::shape::Shape;

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
