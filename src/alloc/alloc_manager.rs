use super::allocatable::WrappedAllocatable;

pub struct AllocManager {
    currently_allocated: Vec<WrappedAllocatable>,
    max_allocated_size: usize,
    currently_allocated_size: usize,
}

impl AllocManager {
    pub fn new(max_allocated_size: usize) -> Self {
        Self {
            currently_allocated: Vec::new(),
            max_allocated_size,
            currently_allocated_size: 0,
        }
    }

    pub fn allocate(&mut self, allocatable: &mut WrappedAllocatable) -> bool {
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

    fn cleanup(&mut self) {
        // cleans up everything that is not longer in use
        let mut indexes_to_clear = Vec::new();
        for (index, allocatable) in self.currently_allocated.iter_mut().enumerate() {
            if !allocatable.is_in_use() {
                indexes_to_clear.push(index);
            }
        }
        for index in indexes_to_clear {
            self.currently_allocated[index].deallocate();
            self.currently_allocated_size -= self.currently_allocated[index].get_size();
            self.currently_allocated.remove(index);
        }
    }

    pub fn get_max_allocated_size(&self) -> usize {
        self.max_allocated_size
    }
}

mod tests {
    use super::*;
    use crate::alloc::allocatable::Allocatable;
    use std::sync::{Arc, Mutex};

    struct TestAllocatable {
        size: usize,
        in_use: bool,
        is_allocated: bool,
    }

    impl TestAllocatable {
        fn new(size: usize) -> Self {
            Self {
                size,
                in_use: false,
                is_allocated: false,
            }
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

    #[test]
    fn test_alloc_manager_only_allocates_once() {
        let mut alloc_manager = AllocManager::new(100);
        let allocatable = WrappedAllocatable::new(Box::new(TestAllocatable::new(50)));
        assert_eq!(alloc_manager.allocate(&mut allocatable.clone()), true);
        assert_eq!(alloc_manager.allocate(&mut allocatable.clone()), false);
    }

    #[test]
    fn test_alloc_manager_gets_through_loop_with_not_much_memory() {
        let mut alloc_manager = AllocManager::new(60);
        let mut allocatable1 = WrappedAllocatable::new(Box::new(TestAllocatable::new(50)));
        let mut allocatable2 = WrappedAllocatable::new(Box::new(TestAllocatable::new(50)));
        let mut allocatable3 = WrappedAllocatable::new(Box::new(TestAllocatable::new(50)));
        let allocatable4 = WrappedAllocatable::new(Box::new(TestAllocatable::new(50)));
        let mut allocatables = vec![
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
            let result = alloc_manager.allocate(&mut this_allocatable.clone());
            assert_eq!(result, true);
            this_allocatable.mark_for_use();
        }
    }
}
