#![warn(clippy::all)]
#![warn(clippy::style)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::cargo)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::multiple_crate_versions)]
pub mod activation;
pub mod layer;
pub mod nn;
pub mod training;
pub mod utilities;
