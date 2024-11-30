use std::{fmt::Error, marker::PhantomData};

use super::{
    challenge::Challenge,
    options::{EvolutionOptions, LogLevel},
};
use crate::evol::evolution::EvolutionResult;
use crate::evol::{phenotype::Phenotype, rng::RandomNumberGenerator, strategy::BreedStrategy};

use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

/// Manages the evolution process using a specified breeding strategy and challenge.
#[derive(Debug, Clone)]
pub struct ParallelEvolutionLauncher<Pheno, Strategy, Chall>
where
    Pheno: Phenotype + Sync + Send,
    Chall: Challenge<Pheno> + Sync + Send + Clone,
    Strategy: BreedStrategy<Pheno> + Sync + Send,
{
    strategy: Arc<Mutex<Strategy>>,
    challenge: Arc<Mutex<Chall>>,
    num_threads: usize,
    _marker: PhantomData<Pheno>,
}

impl<Pheno, Strategy, Chall> ParallelEvolutionLauncher<Pheno, Strategy, Chall>
where
    Pheno: Phenotype + Sync + Send,
    Chall: Challenge<Pheno> + Sync + Send + Clone,
    Strategy: BreedStrategy<Pheno> + Sync + Send,
{
    /// Creates a new `EvolutionLauncher` instance with the specified breeding strategy and challenge.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The breeding strategy used for generating offspring during evolution.
    /// * `challenge` - The challenge used to evaluate the fitness of phenotypes.
    /// * `num_threads` - The number of threads to use for parallel evolution.
    ///
    /// # Returns
    ///
    /// A new `EvolutionLauncher` instance.
    pub fn new(strategy: Strategy, challenge: Chall, num_threads: usize) -> Self {
        Self {
            strategy: Arc::new(Mutex::new(strategy)),
            challenge: Arc::new(Mutex::new(challenge)),
            num_threads,
            _marker: PhantomData,
        }
    }
    /// Evolves a population of phenotypes over multiple generations.
    ///
    /// # Arguments
    ///
    /// * `options` - Evolution options controlling the evolution process.
    /// * `starting_value` - The initial phenotype from which evolution begins.
    /// * `rng` - A random number generator for introducing randomness.
    ///
    /// # Returns
    ///
    /// A `Result` containing the best-evolved phenotype and its associated score, or an `Error` if evolution fails.
    pub fn evolve(
        &self,
        options: &EvolutionOptions,
        starting_value: Pheno,
        rng: &mut RandomNumberGenerator,
    ) -> Result<EvolutionResult<Pheno>, Error> {
        let mut candidates: Vec<Pheno> = Vec::new();
        let fitness: Vec<EvolutionResult<Pheno>> = Vec::new();
        let mut parents: Vec<Pheno> = vec![starting_value];

        // Mutex for safely sharing fitness across threads
        let mutexed_fitness = Mutex::new(fitness);

        // Set up thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build_global()
            .unwrap();

        for generation in 0..options.get_num_generations() {
            candidates.clear();
            candidates.extend(
                self.strategy
                    .lock()
                    .unwrap()
                    .breed(&parents, options, rng)?,
            );

            mutexed_fitness.lock().unwrap().clear();
            let ch = self.challenge.clone();
            candidates.par_iter_mut().for_each(|candidate| {
                let cloned_challenge = ch.lock().unwrap().clone();
                let score = cloned_challenge.score(candidate);
                let result = EvolutionResult {
                    pheno: candidate.clone(),
                    score,
                };

                // Push to the shared fitness vector
                mutexed_fitness.lock().unwrap().push(result);
            });

            mutexed_fitness
                .lock()
                .unwrap()
                .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

            match options.get_log_level() {
                LogLevel::Minimal => println!("Generation: {}", generation),
                LogLevel::Verbose => {
                    mutexed_fitness.lock().unwrap().iter().for_each(|result| {
                        println!("Generation: {} \n", generation);
                        println!("Phenotype: {:?} \n Score: {}", result.pheno, result.score);
                    });
                }
                LogLevel::None => {}
            }

            parents.clear();
            mutexed_fitness
                .lock()
                .unwrap()
                .iter()
                .take(options.get_population_size())
                .for_each(|fitness_result| parents.push(fitness_result.pheno.clone()));
        }
        let winner = mutexed_fitness.lock().unwrap()[0].clone();
        Ok(winner)
    }
}
