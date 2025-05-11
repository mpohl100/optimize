use std::sync::{Arc, Mutex};

use crate::neural::layer::layer_trait::{WrappedLayer, WrappedTrainableLayer};

use super::{layer_alloc::{LayerAllocManager, WrappedLayerAllocManager}, trainable_layer_alloc::{TrainableLayerAllocManager, WrappedTrainableLayerAllocManager}};

#[derive(Default, Debug, Clone)]
pub struct Utils {
    layer_alloc_manager: WrappedLayerAllocManager,
    trainable_layer_alloc_manager: WrappedTrainableLayerAllocManager,
}

impl Utils {
    pub fn new(cpu_memory: usize) -> Self {
        Self {
            layer_alloc_manager: WrappedLayerAllocManager::new(LayerAllocManager::new(cpu_memory)),
            trainable_layer_alloc_manager: WrappedTrainableLayerAllocManager::new(TrainableLayerAllocManager::new(cpu_memory)),
        }
    }

    pub fn allocate(&mut self, allocatable: WrappedLayer) -> bool {
        self.layer_alloc_manager.allocate(allocatable)
    }

    pub fn deallocate(&mut self, allocatable: WrappedLayer) {
        self.layer_alloc_manager.deallocate(allocatable);
    }

    pub fn allocate_trainable(&mut self, allocatable: WrappedTrainableLayer) -> bool {
        self.trainable_layer_alloc_manager.allocate(allocatable)
    }

    pub fn deallocate_trainable(&mut self, allocatable: WrappedTrainableLayer) {
        self.trainable_layer_alloc_manager.deallocate(allocatable);
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

    pub fn allocate(&mut self, allocatable: WrappedLayer) -> bool {
        self.utils.lock().unwrap().allocate(allocatable)
    }

    pub fn get_max_allocated_size(&self) -> usize {
        self.utils.lock().unwrap().get_max_allocated_size()
    }
}
