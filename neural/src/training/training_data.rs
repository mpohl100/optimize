use num_traits::NumCast;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

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

/// Owned storage for training data inputs and targets
#[derive(Debug)]
struct OwnedTrainingData {
    inputs: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
}

/// Thread-safe wrapper around owned training data using `Arc<Mutex<...>>`
#[derive(Clone, Debug)]
pub struct WrappedTrainingData {
    inner: Arc<Mutex<OwnedTrainingData>>,
}

impl WrappedTrainingData {
    /// Creates a new `WrappedTrainingData` from owned inputs and targets.
    ///
    /// # Panics
    ///
    /// Panics if inputs and targets have different lengths.
    #[must_use]
    pub fn new(
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
    ) -> Self {
        assert_eq!(inputs.len(), targets.len(), "Inputs and targets must have the same length");
        Self { inner: Arc::new(Mutex::new(OwnedTrainingData { inputs, targets })) }
    }

    /// Returns the number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        safe_lock(&self.inner).inputs.len()
    }

    /// Returns `true` if there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns cloned inputs and targets for the given index permutation.
    #[must_use]
    pub fn get_by_indices(
        &self,
        indices: &[usize],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let data = safe_lock(&self.inner);
        let inputs = indices.iter().map(|&i| data.inputs[i].clone()).collect();
        let targets = indices.iter().map(|&i| data.targets[i].clone()).collect();
        (inputs, targets)
    }

    /// Returns a clone of all inputs.
    #[must_use]
    pub fn inputs(&self) -> Vec<Vec<f64>> {
        safe_lock(&self.inner).inputs.clone()
    }

    /// Returns a clone of all targets.
    #[must_use]
    pub fn targets(&self) -> Vec<Vec<f64>> {
        safe_lock(&self.inner).targets.clone()
    }
}

/// A view of `WrappedTrainingData` with randomly shuffled sample indices.
///
/// The constructor shuffles the sample indices once; subsequent calls to
/// [`RandomTrainingDataView::get_samples_and_labels`] return subsets based on
/// those fixed random positions.
#[derive(Clone, Debug)]
pub struct RandomTrainingDataView {
    wrapped_data: WrappedTrainingData,
    random_positions: Arc<Vec<usize>>,
}

impl RandomTrainingDataView {
    /// Creates a new `RandomTrainingDataView` by randomly shuffling the sample indices
    /// of the provided `WrappedTrainingData`.
    #[must_use]
    pub fn new(wrapped_data: WrappedTrainingData) -> Self {
        let len = wrapped_data.len();
        let mut positions: Vec<usize> = (0..len).collect();
        positions.shuffle(&mut thread_rng());
        Self { wrapped_data, random_positions: Arc::new(positions) }
    }

    /// Returns a `WrappedTrainingData` containing the samples whose shuffled positions
    /// fall in `[from * len, to * len)`.
    ///
    /// # Panics
    ///
    /// Panics if `from` or `to` are not in `[0.0, 1.0]`, or if `to <= from`.
    #[must_use]
    pub fn get_samples_and_labels(
        &self,
        from: f64,
        to: f64,
    ) -> WrappedTrainingData {
        assert!((0.0..=1.0).contains(&from), "from must be between 0 and 1, got {from}");
        assert!((0.0..=1.0).contains(&to), "to must be between 0 and 1, got {to}");
        assert!(to > from, "to must be greater than from, got from={from} to={to}");

        let len = self.random_positions.len();
        let len_f64: f64 = NumCast::from(len).unwrap_or(0.0);
        let start: usize = NumCast::from((from * len_f64).floor()).unwrap_or(0);
        let end: usize = NumCast::from((to * len_f64).ceil()).unwrap_or(len);
        let end = end.min(len);

        let selected_indices: &[usize] = &self.random_positions[start..end];
        let (inputs, targets) = self.wrapped_data.get_by_indices(selected_indices);
        WrappedTrainingData::new(inputs, targets)
    }
}
