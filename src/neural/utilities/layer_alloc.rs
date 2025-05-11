use std::{
    ptr,
    sync::{Arc, Mutex},
};

use crate::neural::layer::layer_trait::WrappedLayer;

#[derive(Default, Debug, Clone)]
pub struct LayerAllocManager {
    currently_allocated: Vec<WrappedLayer>,
    max_allocated_size: usize,
    currently_allocated_size: usize,
}

impl LayerAllocManager {
    pub fn new(max_allocated_size: usize) -> Self {
        Self {
            currently_allocated: Vec::new(),
            max_allocated_size,
            currently_allocated_size: 0,
        }
    }

    pub fn allocate(&mut self, allocatable: WrappedLayer) -> bool {
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

    fn deallocate(&mut self, allocatable: WrappedLayer) {
        if !allocatable.is_allocated() {
            return;
        }
        allocatable.deallocate();
        self.currently_allocated_size -= allocatable.get_size();
        self.currently_allocated
            .retain(|x| !ptr::eq(x, &allocatable));
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

    pub fn get_max_allocated_size(&self) -> usize {
        self.max_allocated_size
    }
}

#[derive(Default, Debug, Clone)]
pub struct WrappedLayerAllocManager {
    alloc_manager: Arc<Mutex<LayerAllocManager>>,
}

impl WrappedLayerAllocManager {
    pub fn new(alloc_manager: LayerAllocManager) -> Self {
        Self {
            alloc_manager: Arc::new(Mutex::new(alloc_manager)),
        }
    }

    pub fn allocate(&mut self, allocatable: WrappedLayer) -> bool {
        self.alloc_manager.lock().unwrap().allocate(allocatable)
    }

    pub fn deallocate(&mut self, allocatable: WrappedLayer) {
        self.alloc_manager.lock().unwrap().deallocate(allocatable);
    }

    pub fn get_max_allocated_size(&self) -> usize {
        self.alloc_manager.lock().unwrap().get_max_allocated_size()
    }
}
