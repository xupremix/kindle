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
