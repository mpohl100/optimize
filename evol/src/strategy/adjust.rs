//! # `AdjustStrategy`
//!
//! The `AdjustStrategy` struct represents a breeding strategy where the first
//! parent is considered as the winner of the previous generation, and the remaining
//! parents are used to create new individuals through crossover and mutation.
//! Furthermore depending on how well the fitness of the winner has increased compared to the previous generation, the number of mutate calls is adjusted.
//! If the increase was significant the number of mutate calls is decreased, otherwise it is increased.
use super::BreedStrategy;
use crate::phenotype::Phenotype;
use std::{fmt::Error, marker::PhantomData};

pub trait Adjust<Pheno: Phenotype> {
    fn incr_number_mutates(&mut self) -> usize;
    fn decr_number_mutates(&mut self) -> usize;
    fn get_number_mutates(&self) -> usize;
}

/// # `AdjustStrategy`
///
/// The `AdjustStrategy` struct represents a breeding strategy where the first
/// parent is considered as the winner of the previous generation, and the remaining
/// parents are used to create new individuals through crossover and mutation.
/// Furthermore depending on how well the fitness of the winner has increased compared to the previous generation, the number of mutate calls is adjusted.
/// If the increase was significant the number of mutate calls is decreased, otherwise it is increased.
#[derive(Debug, Clone)]
pub struct AdjustStrategy<Pheno>
where
    Pheno: Phenotype + Adjust<Pheno>,
{
    _marker: PhantomData<Pheno>,
}

impl<Pheno> Default for AdjustStrategy<Pheno>
where
    Pheno: Phenotype + Adjust<Pheno>,
{
    fn default() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<Pheno> BreedStrategy<Pheno> for AdjustStrategy<Pheno>
where
    Pheno: Phenotype + Adjust<Pheno>,
{
    /// Breeds offspring from a set of parent phenotypes
    ///
    /// This method uses a winner-takes-all approach, selecting the first parent
    /// as the winner and evolving it to create offspring.
    ///
    /// # Arguments
    ///
    /// * `parents` - A slice of parent phenotypes.
    /// * `evol_options` - Evolution options controlling the breeding process.
    /// * `rng` - A random number generator for introducing randomness.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of offspring phenotypes or an `Error` if breeding fails.
    fn breed(
        &self,
        parents: &[Pheno],
        evol_options: &crate::evolution::options::EvolutionOptions,
        rng: &mut crate::rng::RandomNumberGenerator,
    ) -> Result<Vec<Pheno>, Error> {
        let mut children: Vec<Pheno> = Vec::new();
        let winner_previous_generation = parents[0].clone();
        let mut first_child = winner_previous_generation.clone();
        first_child.incr_number_mutates();
        children.push(first_child);

        parents.iter().skip(1).try_for_each(|parent| -> Result<(), Error> {
            let mut child = winner_previous_generation.clone();
            child.crossover(parent);
            let mut mutated_child = Self::develop(child, rng);
            mutated_child.decr_number_mutates();
            children.push(mutated_child);
            Ok(())
        })?;

        (parents.len()..evol_options.get_num_offspring()).try_for_each(
            |_| -> Result<(), Error> {
                let child = winner_previous_generation.clone();
                let mut mutated_child = Self::develop(child, rng);
                mutated_child.decr_number_mutates();
                children.push(mutated_child);
                Ok(())
            },
        )?;

        Ok(children)
    }
}

impl<Pheno> AdjustStrategy<Pheno>
where
    Pheno: Phenotype + Adjust<Pheno>,
{
    /// Develops a phenotype with consideration of the fitness increase of the previous generation
    ///
    /// # Arguments
    ///
    /// * `pheno` - The initial phenotype to be developed.
    /// * `rng` - A random number generator for introducing randomness.
    ///
    /// # Returns
    ///
    /// A `Result` containing the developed phenotype
    ///
    /// # Details
    ///
    /// This method develops a phenotype.
    /// The development process involves repeated mutations as often as calculated
    /// calculated by the method `calculate_number_of_mutations`.
    fn develop(
        pheno: Pheno,
        rng: &mut crate::rng::RandomNumberGenerator,
    ) -> Pheno {
        let mut phenotype = pheno;
        let number_of_mutations = phenotype.get_number_mutates();
        phenotype.mutate(rng);
        // call mutate number_of_mutations times
        for _ in 0..number_of_mutations {
            phenotype.mutate(rng);
        }
        phenotype
    }
}
