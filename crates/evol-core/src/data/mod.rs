pub mod loader;

pub struct Dataset<const LABELS: usize, TrainImages, TrainLabels, TestImages, TestLabels> {
    train_images: TrainImages,
    train_labels: TrainLabels,
    test_images: TestImages,
    test_labels: TestLabels,
}

impl<const LABELS: usize, TrainImages, TrainLabels, TestImages, TestLabels>
    Dataset<LABELS, TrainImages, TrainLabels, TestImages, TestLabels>
{
    #[inline(always)]
    pub const fn labels() -> usize {
        LABELS
    }

    #[inline(always)]
    pub const fn n_labels(&self) -> usize {
        LABELS
    }

    #[inline(always)]
    pub fn train_images(&self) -> &TrainImages {
        &self.train_images
    }

    #[inline(always)]
    pub fn train_labels(&self) -> &TrainLabels {
        &self.train_labels
    }

    #[inline(always)]
    pub fn test_images(&self) -> &TestImages {
        &self.test_images
    }

    #[inline(always)]
    pub fn test_labels(&self) -> &TestLabels {
        &self.test_labels
    }
}
