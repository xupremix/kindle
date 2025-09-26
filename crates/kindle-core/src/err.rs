pub type KindleResult<Ok> = Result<Ok, KindleError>;

#[derive(thiserror::Error, Debug)]
pub enum KindleError {
    #[error("The index provided `{idx}`, is out bounds for length `{length}`")]
    DimensionOutOfBounds { idx: usize, length: usize },
}
