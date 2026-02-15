/// Settings for neural network training
#[derive(Clone, Debug)]
pub struct TrainingSettings {
    learning_rate: f64,
    epochs: usize,
    tolerance: f64,
    use_adam: bool,
    validation_split: f64,
    sample_match_percentage: f64,
    batch_size: usize,
}

impl TrainingSettings {
    /// Creates a new `TrainingSettings` instance
    #[must_use]
    pub const fn new(
        learning_rate: f64,
        epochs: usize,
        tolerance: f64,
        use_adam: bool,
        validation_split: f64,
        sample_match_percentage: f64,
        batch_size: usize,
    ) -> Self {
        Self {
            learning_rate,
            epochs,
            tolerance,
            use_adam,
            validation_split,
            sample_match_percentage,
            batch_size,
        }
    }

    /// Returns the learning rate
    #[must_use]
    pub const fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the number of epochs
    #[must_use]
    pub const fn epochs(&self) -> usize {
        self.epochs
    }

    /// Returns the tolerance threshold
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Returns whether to use Adam optimizer
    #[must_use]
    pub const fn use_adam(&self) -> bool {
        self.use_adam
    }

    /// Returns the validation split ratio
    #[must_use]
    pub const fn validation_split(&self) -> f64 {
        self.validation_split
    }

    /// Returns the sample match percentage threshold
    #[must_use]
    pub const fn sample_match_percentage(&self) -> f64 {
        self.sample_match_percentage
    }

    /// Returns the batch size for batch training
    #[must_use]
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
}

impl Default for TrainingSettings {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 1000,
            tolerance: 0.1,
            use_adam: false,
            validation_split: 0.7,
            sample_match_percentage: 1.0,
            batch_size: 32,
        }
    }
}
