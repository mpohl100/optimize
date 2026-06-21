use crate::persist::cpu_matrix::PersistableMatrix;
use crate::persist::traits::PersistableMatrixTrait;
use crate::persist::traits::PersistableValue;
use alloc::allocatable::WrappedAllocatableTrait;

use std::sync::Arc;
use std::sync::Mutex;

use utils::safer::safe_lock;

#[derive(Debug, Clone)]
pub struct WrappedPersistableMatrix<T: PersistableValue + From<f64> + 'static> {
    pm: Arc<Mutex<Box<dyn PersistableMatrixTrait<T>>>>,
}

impl<T: PersistableValue + From<f64> + std::fmt::Debug + 'static> Default
    for WrappedPersistableMatrix<T>
{
    fn default() -> Self {
        Self { pm: Arc::new(Mutex::new(Box::new(PersistableMatrix::<T>::default()))) }
    }
}

#[allow(clippy::fallible_impl_from)]
impl<T: PersistableValue + From<f64> + 'static> From<f64> for WrappedPersistableMatrix<T> {
    fn from(_value: f64) -> Self {
        panic!("Cannot convert f64 to WrappedPersistableMatrix");
    }
}

impl<T: PersistableValue + From<f64> + 'static> WrappedPersistableMatrix<T> {
    #[must_use]
    pub fn new(pm: Box<dyn PersistableMatrixTrait<T>>) -> Self {
        Self { pm: Arc::new(Mutex::new(pm)) }
    }

    #[must_use]
    pub fn get_unchecked(
        &self,
        x: usize,
        y: usize,
    ) -> T {
        let pm = safe_lock(&self.pm);
        pm.get_unchecked(x, y)
    }

    pub fn set_mut_unchecked(
        &self,
        x: usize,
        y: usize,
        value: T,
    ) {
        let mut pm = safe_lock(&self.pm);
        pm.set_mut_unchecked(x, y, value);
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        let pm = safe_lock(&self.pm);
        pm.rows()
    }

    #[must_use]
    pub fn cols(&self) -> usize {
        let pm = safe_lock(&self.pm);
        pm.cols()
    }

    #[must_use]
    pub fn mat(&self) -> Arc<Mutex<Box<dyn PersistableMatrixTrait<T>>>> {
        self.pm.clone()
    }

    /// Save the matrix to disk
    /// # Errors
    /// Returns an error if saving fails
    pub fn save(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut pm = safe_lock(&self.pm);
        pm.save()
    }
}

impl<T: PersistableValue + From<f64> + 'static> WrappedAllocatableTrait
    for WrappedPersistableMatrix<T>
{
    fn allocate(&self) {
        let mut pm = safe_lock(&self.pm);
        pm.allocate();
    }

    fn deallocate(&self) {
        let mut pm = safe_lock(&self.pm);
        pm.deallocate();
    }

    fn is_allocated(&self) -> bool {
        let pm = safe_lock(&self.pm);
        pm.is_allocated()
    }

    fn get_size(&self) -> usize {
        let pm = safe_lock(&self.pm);
        pm.get_size()
    }

    fn mark_for_use(&mut self) {
        let mut pm = safe_lock(&self.pm);
        pm.mark_for_use();
    }

    fn free_from_use(&mut self) {
        let mut pm = safe_lock(&self.pm);
        pm.free_from_use();
    }

    fn is_in_use(&self) -> bool {
        let pm = safe_lock(&self.pm);
        pm.is_in_use()
    }
}
