use utils::safer::safe_lock;

use std::sync::{Arc, Mutex};

#[derive(Default, Debug, Clone)]
pub struct MemoryCounter {
    current_memory: usize,
    capacity_memory: usize,
}

impl MemoryCounter {
    #[must_use]
    pub const fn new(capacity_memory: usize) -> Self {
        Self { current_memory: 0, capacity_memory }
    }

    /// Try to allocate memory of size `size`.
    /// Returns true if successful, false if not enough memory.
    pub fn try_allocate(
        &mut self,
        size: usize,
    ) -> bool {
        if self.current_memory + size <= self.capacity_memory {
            self.current_memory += size;
            true
        } else {
            false
        }
    }

    /// Free memory of size `size`.
    pub fn free(
        &mut self,
        size: usize,
    ) {
        if size > self.current_memory {
            self.current_memory = 0;
        } else {
            self.current_memory -= size;
        }
    }

    #[must_use]
    pub const fn get_current_memory(&self) -> usize {
        self.current_memory
    }

    #[must_use]
    pub const fn get_capacity_memory(&self) -> usize {
        self.capacity_memory
    }
}

#[derive(Debug, Default, Clone)]
pub struct WrappedMemoryCounter {
    memory_counter: Arc<Mutex<MemoryCounter>>,
}

impl WrappedMemoryCounter {
    #[must_use]
    pub fn new(memory_counter: MemoryCounter) -> Self {
        Self { memory_counter: Arc::new(Mutex::new(memory_counter)) }
    }

    /// Try to allocate memory of size `size`.
    /// Returns true if successful, false if not enough memory.
    #[must_use]
    pub fn try_allocate(
        &self,
        size: usize,
    ) -> bool {
        let mut mc = safe_lock(&self.memory_counter);
        mc.try_allocate(size)
    }

    /// Free memory of size `size`.
    pub fn free(
        &self,
        size: usize,
    ) {
        let mut mc = safe_lock(&self.memory_counter);
        mc.free(size);
    }

    #[must_use]
    pub fn get_current_memory(&self) -> usize {
        let mc = safe_lock(&self.memory_counter);
        mc.get_current_memory()
    }

    #[must_use]
    pub fn get_capacity_memory(&self) -> usize {
        let mc = safe_lock(&self.memory_counter);
        mc.get_capacity_memory()
    }
}
