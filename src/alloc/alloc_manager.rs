use super::allocatable::WrappedAllocatableTrait;

use std::{
    ptr,
    sync::{Arc, Mutex},
};

#[derive(Default, Debug, Clone)]
pub struct AllocManager<WrappedType: WrappedAllocatableTrait> {
    currently_allocated: Vec<WrappedType>,
    max_allocated_size: usize,
    currently_allocated_size: usize,
}

impl<WrappedType: WrappedAllocatableTrait> AllocManager<WrappedType> {
    #[must_use] pub const fn new(max_allocated_size: usize) -> Self {
        Self { currently_allocated: Vec::new(), max_allocated_size, currently_allocated_size: 0 }
    }

    pub fn allocate(
        &mut self,
        allocatable: WrappedType,
    ) -> bool {
        if allocatable.is_allocated() {
            return false;
        }
        if self.currently_allocated_size + allocatable.get_size() <= self.max_allocated_size {
            // a not yet allocated allocatable can not yet be in use
            // that is why one does not need to check if it is in use
            allocatable.allocate();
            self.currently_allocated.push(allocatable.clone());
            self.currently_allocated_size += allocatable.get_size();
            return true;
        } else {
            // too much is allocated, try cleaning up and to then do the allocation
            self.cleanup();
            if self.currently_allocated_size + allocatable.get_size() <= self.max_allocated_size {
                // a not yet allocated allocatable can not yet be in use
                // that is why one does not need to check if it is in use
                allocatable.allocate();
                self.currently_allocated.push(allocatable.clone());
                self.currently_allocated_size += allocatable.get_size();
                return true;
            }
        }
        false
    }

    fn deallocate(
        &mut self,
        allocatable: WrappedType,
    ) {
        if !allocatable.is_allocated() {
            return;
        }
        allocatable.deallocate();
        self.currently_allocated_size -= allocatable.get_size();
        self.currently_allocated.retain(|x| !ptr::eq(x, &allocatable));
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
            self.deallocate(self.currently_allocated[index].clone());
        }
    }

    #[must_use] pub const fn get_max_allocated_size(&self) -> usize {
        self.max_allocated_size
    }
}

#[derive(Default, Debug, Clone)]
pub struct WrappedAllocManager<WrappedType: WrappedAllocatableTrait> {
    alloc_manager: Arc<Mutex<AllocManager<WrappedType>>>,
}

impl<WrappedType: WrappedAllocatableTrait> WrappedAllocManager<WrappedType> {
    #[must_use] pub fn new(alloc_manager: AllocManager<WrappedType>) -> Self {
        Self { alloc_manager: Arc::new(Mutex::new(alloc_manager)) }
    }

    pub fn allocate(
        &mut self,
        allocatable: WrappedType,
    ) -> bool {
        self.alloc_manager.lock().unwrap().allocate(allocatable)
    }

    pub fn deallocate(
        &mut self,
        allocatable: WrappedType,
    ) {
        self.alloc_manager.lock().unwrap().deallocate(allocatable);
    }

    #[must_use] pub fn get_max_allocated_size(&self) -> usize {
        self.alloc_manager.lock().unwrap().get_max_allocated_size()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use crate::alloc::alloc_manager::AllocManager;
    use crate::alloc::allocatable::Allocatable;
    use crate::alloc::allocatable::WrappedAllocatableTrait;

    struct TestAllocatable {
        size: usize,
        in_use: bool,
        is_allocated: bool,
    }

    impl TestAllocatable {
        fn new(size: usize) -> Self {
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
        let mut alloc_manager = AllocManager::new(100);
        let allocatable = WrappedTestAllocatable::new(TestAllocatable::new(50));
        assert!(alloc_manager.allocate(allocatable.clone()));
        assert!(!alloc_manager.allocate(allocatable.clone()));
    }

    #[test]
    fn test_alloc_manager_gets_through_loop_with_not_much_memory() {
        let mut alloc_manager = AllocManager::new(60);
        let mut allocatable1 = WrappedTestAllocatable::new(TestAllocatable::new(50));
        let mut allocatable2 = WrappedTestAllocatable::new(TestAllocatable::new(50));
        let mut allocatable3 = WrappedTestAllocatable::new(TestAllocatable::new(50));
        let allocatable4 = WrappedTestAllocatable::new(TestAllocatable::new(50));
        let mut allocatables = [
            allocatable1.clone(),
            allocatable2.clone(),
            allocatable3.clone(),
            allocatable4.clone(),
        ];
        for (index, this_allocatable) in allocatables.iter_mut().enumerate() {
            if index == 1 {
                allocatable1.free_from_use();
            } else if index == 2 {
                allocatable2.free_from_use();
            } else if index == 3 {
                allocatable3.free_from_use();
            }
            let result = alloc_manager.allocate(this_allocatable.clone());
            assert!(result);
            this_allocatable.mark_for_use();
        }
    }
}
