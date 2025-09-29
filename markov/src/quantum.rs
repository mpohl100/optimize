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

    /// Set the expected energy and standard deviation for the quantum roll
    ///
    /// # Panics
    /// Panics if the standard deviation is negative
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn roll(
        &mut self,
        state: &State,
    ) -> f64 {
        let index = self.states.iter().position(|s| *s == *state);
        if let Some(_i) = index {
            // find the state in the already_rolled map
            if let Some(value) = self.already_rolled.get(state) {
                *value
            } else {
                let normal_distribution =
                    Normal::new(self.expected_energy, self.standard_deviation).unwrap();
                let energy = normal_distribution.sample(&mut rand::thread_rng());
                self.set_energy(state, energy);
                energy
            }
        } else {
            panic!("State not found in the list of states");
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn set_energy(
        &mut self,
        state: &State,
        energy: f64,
    ) {
        // if the energy is already set, early return
        if self.already_rolled.contains_key(state) {
            return;
        }
        self.already_rolled.insert(state.clone(), energy);
        let index = self.states.iter().position(|s| *s == *state).unwrap();
        let mut connected_states = Vec::new();
        for j in 0..self.entanglement_matrix.cols() {
            let val = self.entanglement_matrix.get(index, j).unwrap();
            if val != 0.0 {
                connected_states.push(self.states[j].clone());
            }
        }
        // remove the ith state from the connected states
        connected_states.retain(|s| *s != *state);
        let relative_energy = -(energy - self.expected_energy) / self.standard_deviation;
        let distributed_energy = relative_energy / (connected_states.len() as f64);
        for connected_state in connected_states {
            self.set_energy(&connected_state, distributed_energy);
        }
    }
}
