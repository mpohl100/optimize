use std::collections::BTreeMap;

// Quantum module for Markov decision processes
use crate::solver::StateTrait;
use matrix::mat::WrappedMatrix;
use rand_distr::{Distribution, Normal};

#[derive(Clone)]
pub struct QuantumEnergyRoller<State: StateTrait + 'static> {
    states: Vec<State>,
    entanglement_matrix: WrappedMatrix<f64>,
    expected_energy: f64,
    standard_deviation: f64,
    already_rolled: BTreeMap<State, f64>,
}

impl<State: StateTrait + 'static> QuantumEnergyRoller<State> {
    #[must_use]
    pub const fn new(
        states: Vec<State>,
        entanglement_matrix: WrappedMatrix<f64>,
    ) -> Self {
        Self {
            states,
            entanglement_matrix,
            expected_energy: 0.0,
            standard_deviation: 0.0,
            already_rolled: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn roll(
        &self,
        state: State,
    ) -> f64 {
        let index = self.states.iter().position(|s| *s == state);
        if let Some(i) = index {
            // find the state in the already_rolled map
            if let Some(value) = self.already_rolled.get(&state) {
                *value
            } else {
                let normal_distribution =
                    Normal::new(self.expected_energy, self.standard_deviation).unwrap();
                let energy = normal_distribution.sample(&mut rand::thread_rng());
                energy
            }
        } else {
            panic!("State not found in the list of states");
        }
    }
}
