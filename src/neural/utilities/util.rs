use std::sync::{Arc, Mutex};

use crate::neural::layer::layer_trait::WrappedLayer;

use super::layer_alloc::{LayerAllocManager, WrappedLayerAllocManager};


#[derive(Default, Debug, Clone)]
pub struct Utils{
    pub layer_alloc_manager: WrappedLayerAllocManager,
}

impl Utils {
    pub fn new(cpu_memory: usize) -> Self {
        Self {
            layer_alloc_manager: WrappedLayerAllocManager::new(LayerAllocManager::new(cpu_memory)),
        }
    }

    pub fn allocate(&mut self, allocatable: &mut WrappedLayer) -> bool {
        self.layer_alloc_manager.allocate(allocatable)
    }

    pub fn get_max_allocated_size(&self) -> usize {
        self.layer_alloc_manager.get_max_allocated_size()
    }
}

#[derive(Default, Debug, Clone)]
pub struct WrappedUtils {
    utils: Arc<Mutex<Utils>>,
}

impl WrappedUtils {
    pub fn new(utils: Utils) -> Self {
        Self {
            utils: Arc::new(Mutex::new(utils)),
        }
    }

    pub fn allocate(&mut self, allocatable: &mut WrappedLayer) -> bool {
        self.utils.lock().unwrap().allocate(allocatable)
    }
    
    pub fn get_max_allocated_size(&self) -> usize {
        self.utils.lock().unwrap().get_max_allocated_size()
    }
}