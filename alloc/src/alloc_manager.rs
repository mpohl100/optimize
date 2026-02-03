use crate::memory_counter::WrappedMemoryCounter;

use super::allocatable::WrappedAllocatableTrait;

use utils::safer::safe_lock;

use std::{
    ptr,
    sync::{Arc, Mutex},
};

#[derive(Default, Debug, Clone)]
pub struct AllocManager<WrappedType: WrappedAllocatableTrait> {
    currently_allocated: Vec<WrappedType>,
    memory_counter: WrappedMemoryCounter,
}

impl<WrappedType: WrappedAllocatableTrait> AllocManager<WrappedType> {
    #[must_use]
    pub const fn new(memory_counter: WrappedMemoryCounter) -> Self {
        Self { currently_allocated: Vec::new(), memory_counter }
    }

    pub fn allocate(
        &mut self,
        allocatable: &WrappedType,
    ) -> bool {
        if allocatable.is_allocated() {
            return false;
        }
        if self.memory_counter.get_current_memory() + allocatable.get_size()
            <= self.memory_counter.get_capacity_memory()
        {
            // a not yet allocated allocatable can not yet be in use
            // that is why one does not need to check if it is in use
            allocatable.allocate();
            self.currently_allocated.push(allocatable.clone());
            let _ = self.memory_counter.try_allocate(allocatable.get_size());
            return true;
        }
        // too much is allocated, try cleaning up and to then do the allocation
        self.cleanup();
        if self.memory_counter.get_current_memory() + allocatable.get_size()
            <= self.memory_counter.get_capacity_memory()
        {
            // a not yet allocated allocatable can not yet be in use
            // that is why one does not need to check if it is in use
            allocatable.allocate();
            self.currently_allocated.push(allocatable.clone());
            let _ = self.memory_counter.try_allocate(allocatable.get_size());
            return true;
        }
        false
    }

    fn deallocate(
        &mut self,
        allocatable: &WrappedType,
    ) {
        if !allocatable.is_allocated() {
            return;
        }
        allocatable.deallocate();
        self.memory_counter.free(allocatable.get_size());
        self.currently_allocated.retain(|x| !ptr::eq(x, allocatable));
    }

    fn cleanup(&mut self) {
        // cleans up everything that is not longer in use
        let mut indexes_to_clear = Vec::new();
        for (index, allocatable) in self.currently_allocated.iter_mut().enumerate() {
            if !allocatable.is_in_use() {
                indexes_to_clear.push(index);
            }
        }
        for index in indexes_to_clear {
            let currently_allocated = self.currently_allocated[index].clone();
            self.deallocate(&currently_allocated);
        }
    }

    #[must_use]
    pub fn get_max_allocated_size(&self) -> usize {
        self.memory_counter.get_capacity_memory()
    }
}

#[derive(Default, Debug, Clone)]
pub struct WrappedAllocManager<WrappedType: WrappedAllocatableTrait> {
    alloc_manager: Arc<Mutex<AllocManager<WrappedType>>>,
}

impl<WrappedType: WrappedAllocatableTrait> WrappedAllocManager<WrappedType> {
    #[must_use]
    pub fn new(alloc_manager: AllocManager<WrappedType>) -> Self {
        Self { alloc_manager: Arc::new(Mutex::new(alloc_manager)) }
    }

    /// Attempts to allocate the given allocatable object.
    ///
    /// # Panics
    /// Panics if the mutex guarding the underlying `AllocManager` is poisoned.
    pub fn allocate(
        &mut self,
        allocatable: &WrappedType,
    ) -> bool {
        self.alloc_manager.lock().unwrap().allocate(allocatable)
    }

    /// Deallocates the given allocatable object.
    ///
    /// # Panics
    /// Panics if the mutex guarding the underlying `AllocManager` is poisoned.
    pub fn deallocate(
        &mut self,
        allocatable: &WrappedType,
    ) {
        self.alloc_manager.lock().unwrap().deallocate(allocatable);
    }

    #[must_use]
    pub fn get_max_allocated_size(&self) -> usize {
        safe_lock(&self.alloc_manager).get_max_allocated_size()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use crate::alloc_manager::AllocManager;
    use crate::allocatable::Allocatable;
    use crate::allocatable::WrappedAllocatableTrait;
    use crate::memory_counter::MemoryCounter;
    use crate::memory_counter::WrappedMemoryCounter;

    struct TestAllocatable {
        size: usize,
        in_use: bool,
        is_allocated: bool,
    }

    impl TestAllocatable {
        const fn new(size: usize) -> Self {
            Self { size, in_use: false, is_allocated: false }
        }
    }

    impl Allocatable for TestAllocatable {
        fn allocate(&mut self) {
            self.is_allocated = true;
        }

        fn deallocate(&mut self) {
            self.is_allocated = false;
        }

        fn is_allocated(&self) -> bool {
            self.is_allocated
        }

        fn get_size(&self) -> usize {
            self.size
        }

        fn mark_for_use(&mut self) {
            self.in_use = true;
        }

        fn free_from_use(&mut self) {
            self.in_use = false;
        }

        fn is_in_use(&self) -> bool {
            self.in_use
        }
    }

    #[derive(Clone)]
    struct WrappedTestAllocatable {
        allocatable: Arc<Mutex<TestAllocatable>>,
    }

    impl WrappedTestAllocatable {
        pub fn new(allocatable: TestAllocatable) -> Self {
            Self { allocatable: Arc::new(Mutex::new(allocatable)) }
        }
    }

    impl WrappedAllocatableTrait for WrappedTestAllocatable {
        fn allocate(&self) {
            self.allocatable.lock().unwrap().allocate();
        }

        fn deallocate(&self) {
            self.allocatable.lock().unwrap().deallocate();
        }

        fn is_allocated(&self) -> bool {
            self.allocatable.lock().unwrap().is_allocated()
        }

        fn get_size(&self) -> usize {
            self.allocatable.lock().unwrap().get_size()
        }

        fn mark_for_use(&mut self) {
            self.allocatable.lock().unwrap().mark_for_use();
        }

        fn free_from_use(&mut self) {
            self.allocatable.lock().unwrap().free_from_use();
        }

        fn is_in_use(&self) -> bool {
            self.allocatable.lock().unwrap().is_in_use()
        }
    }

    #[test]
    fn test_alloc_manager_only_allocates_once() {
        let mut alloc_manager =
            AllocManager::new(WrappedMemoryCounter::new(MemoryCounter::new(100)));
        let allocatable = WrappedTestAllocatable::new(TestAllocatable::new(50));
        assert!(alloc_manager.allocate(&allocatable));
        assert!(!alloc_manager.allocate(&allocatable));
    }

    #[test]
    fn test_alloc_manager_gets_through_loop_with_not_much_memory() {
        let mut alloc_manager =
            AllocManager::new(WrappedMemoryCounter::new(MemoryCounter::new(60)));
        let mut allocatable1 = WrappedTestAllocatable::new(TestAllocatable::new(50));
        let mut allocatable2 = WrappedTestAllocatable::new(TestAllocatable::new(50));
        let mut allocatable3 = WrappedTestAllocatable::new(TestAllocatable::new(50));
        let allocatable4 = WrappedTestAllocatable::new(TestAllocatable::new(50));
        let mut multiple_allocatables =
            [allocatable1.clone(), allocatable2.clone(), allocatable3.clone(), allocatable4];
        for (index, this_allocatable) in multiple_allocatables.iter_mut().enumerate() {
            if index == 1 {
                allocatable1.free_from_use();
            } else if index == 2 {
                allocatable2.free_from_use();
            } else if index == 3 {
                allocatable3.free_from_use();
            }
            let result = alloc_manager.allocate(this_allocatable);
            assert!(result);
            this_allocatable.mark_for_use();
        }
    }
}
