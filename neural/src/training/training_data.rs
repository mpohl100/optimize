/// Training data containing inputs and targets for neural network training
#[derive(Clone, Debug)]
pub struct TrainingData<'a> {
    inputs: &'a [Vec<f64>],
    targets: &'a [Vec<f64>],
}

impl<'a> TrainingData<'a> {
    /// Creates a new `TrainingData` instance
    ///
    /// # Panics
    ///
    /// Panics if inputs and targets have different lengths
    #[must_use]
    pub fn new(
        inputs: &'a [Vec<f64>],
        targets: &'a [Vec<f64>],
    ) -> Self {
        assert_eq!(inputs.len(), targets.len(), "Inputs and targets must have the same length");
        Self { inputs, targets }
    }

    /// Returns a reference to the input data
    #[must_use]
    pub const fn inputs(&self) -> &[Vec<f64>] {
        self.inputs
    }

    /// Returns a reference to the target data
    #[must_use]
    pub const fn targets(&self) -> &[Vec<f64>] {
        self.targets
    }
}
