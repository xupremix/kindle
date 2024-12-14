use crate::{
    prelude::Chunk,
    tensor::{Device, Kind, Shape, Tensor},
};

impl<S: Shape, K: Kind, D: Device> Tensor<S, K, D> {
    // Here instead of requiring the number of chunks
    // we require the number of elements in each chunk
    // such that we can generate the shape and not depend on the
    // user to provide it
    pub fn chunk<const DIM: usize, const NELEMS: usize>(&self) -> Vec<Tensor<S::ChunkShape, K, D>>
    where
        S: Chunk<DIM, NELEMS>,
    {
        S::CHUNK_CHECK;
        self.repr
            .chunk(S::dims()[DIM] / NELEMS, DIM)
            .unwrap()
            .into_iter()
            .map(|repr| Tensor {
                repr,
                ..Default::default()
            })
            .collect()
    }
}
