use std::error::Error;

pub trait PersistableValue: Default + Clone {
    fn to_string_for_matrix(&self) -> String;
    /// # Errors
    /// Returns an error if the string cannot be parsed into the type.
    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;
}
