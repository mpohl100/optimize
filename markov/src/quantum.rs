use std::collections::BTreeMap;

// Quantum module for Markov decision processes
use crate::solver::{ExpectedValueFuncWrapper, MarkovSolver, StateTrait};
use matrix::mat::WrappedMatrix;
use rand_distr::{Distribution, Normal};

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct WrappedQuantumEnergyRoller<State: StateTrait + Send + Sync + 'static> {
    roller: Arc<Mutex<QuantumEnergyRoller<State>>>,
}

impl<State: StateTrait + Send + Sync + 'static> WrappedQuantumEnergyRoller<State> {
    #[must_use]
    pub fn new(roller: QuantumEnergyRoller<State>) -> Self {
        Self { roller: Arc::new(Mutex::new(roller)) }
    }

    /// Roll the quantum energy roller for a given state
    /// If `only_consider_dependent_states` is true, only the entangled states
    /// will be considered for energy distribution
    ///
    /// # Panics
    /// Panics if the mutex is poisoned
    #[must_use]
    pub fn roll(
        &self,
        state: &State,
        only_consider_dependent_states: bool,
    ) -> f64 {
        let mut roller = self.roller.lock().unwrap();
        roller.roll(state, only_consider_dependent_states)
    }

    /// Reset the roller, clearing all previously rolled energies
    /// # Panics
    /// Panics if the mutex is poisoned
    pub fn reset(&self) {
        let mut roller = self.roller.lock().unwrap();
        roller.reset();
    }
}

pub struct QuantumSolver<State: StateTrait + Send + Sync + 'static> {
    roller: WrappedQuantumEnergyRoller<State>,
    total_transition_solver: MarkovSolver<State>,
    entangled_transition_solver: MarkovSolver<State>,
}

impl<State: StateTrait + Send + Sync + 'static> QuantumSolver<State> {
    /// Create a new `QuantumSolver` with the given `QuantumEnergyRoller`
    /// This will create two `MarkovSolvers`:
    /// one for total energy calculation and one for entangled energy calculation
    /// # Panics
    /// Panics if the mutex is poisoned
    #[must_use]
    pub fn new(roller: QuantumEnergyRoller<State>) -> Self {
        let wrapped_roller = Arc::new(WrappedQuantumEnergyRoller::new(roller));
        let total_energy_func_roller = Arc::clone(&wrapped_roller);
        let total_energy_func = move |s: &State| -> f64 { total_energy_func_roller.roll(s, false) };
        let total_energy_func =
            ExpectedValueFuncWrapper::new(Arc::new(Mutex::new(total_energy_func)));
        let total_transition_solver = MarkovSolver::new(
            Arc::clone(&wrapped_roller).roller.lock().unwrap().states.clone(),
            total_energy_func,
        );
        let entangled_energy_func_roller = Arc::clone(&wrapped_roller);
        let entangled_energy_func =
            move |s: &State| -> f64 { entangled_energy_func_roller.roll(s, true) };
        let entangled_energy_func =
            ExpectedValueFuncWrapper::new(Arc::new(Mutex::new(entangled_energy_func)));
        let entangled_transition_solver = MarkovSolver::new(
            Arc::clone(&wrapped_roller).roller.lock().unwrap().states.clone(),
            entangled_energy_func,
        );
        Self {
            roller: Arc::try_unwrap(wrapped_roller).unwrap(),
            total_transition_solver,
            entangled_transition_solver,
        }
    }

    pub fn solve(
        &mut self,
        iterations: usize,
    ) {
        self.total_transition_solver.solve(iterations);
        self.entangled_transition_solver.solve(iterations);
    }
}
