
use std::sync::{Arc, Mutex};

pub trait Allocatable {
    fn allocate(&mut self);
    fn deallocate(&mut self);
    fn is_allocated(&self) -> bool;
    fn get_size(&self) -> usize;
    fn mark_for_use(&mut self);
    fn free_from_use(&mut self);
    fn is_in_use(&self) -> bool;
}

#[derive(Clone)]
pub struct WrappedAllocatable {
    allocatable: Arc<Mutex<Box<dyn Allocatable>>>,
}

impl WrappedAllocatable {
    pub fn new(allocatable: Box<dyn Allocatable>) -> Self {
        Self {
            allocatable: Arc::new(Mutex::new(allocatable)),
        }
    }

    pub fn allocate(&mut self) {
        self.allocatable.lock().unwrap().allocate();
    }

    pub fn deallocate(&mut self) {
        self.allocatable.lock().unwrap().deallocate();
    }

    pub fn is_allocated(&self) -> bool {
        self.allocatable.lock().unwrap().is_allocated()
    }

    pub fn get_size(&self) -> usize {
        self.allocatable.lock().unwrap().get_size()
    }

    pub fn mark_for_use(&mut self) {
        self.allocatable.lock().unwrap().mark_for_use()
    }

    pub fn free_from_use(&mut self) {
        self.allocatable.lock().unwrap().free_from_use()
    }

    pub fn is_in_use(&self) -> bool {
        self.allocatable.lock().unwrap().is_in_use()
    }
}