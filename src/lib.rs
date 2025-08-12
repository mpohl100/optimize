#![warn(clippy::all)]
#![warn(clippy::style)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![warn(clippy::nursery)]
#![warn(clippy::cargo)]
#[allow(clippy::multiple_crate_versions)]
pub mod alloc;
pub mod evol;
pub mod gen;
pub mod neural;
pub mod regret;
