use std::error::Error;
use std::fmt::Debug;

use alloc::allocatable::Allocatable;

use crate::mat::WrappedMatrix;

pub trait PersistableValue: Default + Clone {
    fn to_string_for_matrix(&self) -> String;
    /// # Errors
    /// Returns an error if the string cannot be parsed into the type.
    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;
}

pub trait PersistableMatrixTrait<T: PersistableValue + From<f64> + 'static>:
    Debug + Allocatable
{
    fn get_unchecked(
        &self,
        x: usize,
        y: usize,
    ) -> T;
    fn set_mut_unchecked(
        &mut self,
        x: usize,
        y: usize,
        value: T,
    );

    fn mat(&self) -> Option<&WrappedMatrix<T>>;
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// Save the matrix to disk
    /// # Errors
    /// Returns an error if saving fails
    fn save(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}
