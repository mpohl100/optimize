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

    pub fn reset(&mut self) {
        self.already_rolled.clear();
    }

    /// Entangle two states together
    /// This creates a quantum link between the two states, allowing for
    /// energy to be shared between them during a roll
    ///
    /// # Panics
    /// Panics if either state is not in the list of states
    ///
    pub fn entangle_states(
        &mut self,
        state: &State,
        other_state: &State,
    ) {
        let index = self.states.iter().position(|s| *s == *state).unwrap();
        let other_index = self.states.iter().position(|s| *s == *other_state).unwrap();
        self.entanglement_matrix.set_mut_unchecked(index, other_index, 1.0);
        self.entanglement_matrix.set_mut_unchecked(other_index, index, 1.0);
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
        only_consider_dependent_states: bool,
    ) -> f64 {
        // if the energy is already set, return it and remove the state from the map
        if self.already_rolled.contains_key(state) {
            return self.already_rolled.remove(state).unwrap();
        }
        let normal = Normal::new(self.expected_energy, self.standard_deviation)
            .expect("Standard deviation must be non-negative");
        let energy = normal.sample(&mut rand::thread_rng());
        self.set_energy(state, energy);
        if only_consider_dependent_states {
            0.0
        } else {
            self.already_rolled.remove(state).unwrap()
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
