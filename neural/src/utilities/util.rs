use crate::neural::utilities::safer::safe_lock;

use std::sync::{Arc, Mutex};

use crate::{
    alloc::alloc_manager::{AllocManager, WrappedAllocManager},
    neural::layer::layer_trait::{WrappedLayer, WrappedTrainableLayer},
};

use indicatif::MultiProgress;
use rayon::ThreadPoolBuilder;

#[derive(Debug, Clone)]
pub struct WrappedThreadPool {
    thread_pool: Arc<Mutex<rayon::ThreadPool>>,
}

impl WrappedThreadPool {
    /// Creates a new `WrappedThreadPool` with the specified number of threads.
    ///
    /// # Panics
    /// Panics if the thread pool cannot be built.
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
        Self { thread_pool: Arc::new(Mutex::new(thread_pool)) }
    }

    pub fn execute<F, R>(
        &self,
        f: F,
    ) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        safe_lock(&self.thread_pool).install(f)
    }
}

#[derive(Debug, Clone)]
pub struct Utils {
    layer_alloc_manager: WrappedAllocManager<WrappedLayer>,
    trainable_layer_alloc_manager: WrappedAllocManager<WrappedTrainableLayer>,
    mutli_progress: Arc<MultiProgress>,
    thread_pool: WrappedThreadPool,
}

impl Utils {
    #[must_use]
    pub fn new(
        cpu_memory: usize,
        num_threads: usize,
    ) -> Self {
        Self {
            layer_alloc_manager: WrappedAllocManager::<WrappedLayer>::new(AllocManager::<
                WrappedLayer,
            >::new(
                cpu_memory
            )),
            trainable_layer_alloc_manager: WrappedAllocManager::<WrappedTrainableLayer>::new(
                AllocManager::<WrappedTrainableLayer>::new(cpu_memory),
            ),
            mutli_progress: Arc::new(MultiProgress::new()),
            thread_pool: WrappedThreadPool::new(num_threads),
        }
    }

    pub fn allocate(
        &mut self,
        allocatable: &WrappedLayer,
    ) -> bool {
        self.layer_alloc_manager.allocate(allocatable)
    }

    pub fn deallocate(
        &mut self,
        allocatable: &WrappedLayer,
    ) {
        self.layer_alloc_manager.deallocate(allocatable);
    }

    pub fn allocate_trainable(
        &mut self,
        allocatable: &WrappedTrainableLayer,
    ) -> bool {
        self.trainable_layer_alloc_manager.allocate(allocatable)
    }

    pub fn deallocate_trainable(
        &mut self,
        allocatable: &WrappedTrainableLayer,
    ) {
        self.trainable_layer_alloc_manager.deallocate(allocatable);
    }

    #[must_use]
    pub fn get_max_allocated_size(&self) -> usize {
        self.layer_alloc_manager.get_max_allocated_size()
    }

    #[must_use]
    pub fn get_multi_progress(&self) -> Arc<MultiProgress> {
        self.mutli_progress.clone()
    }

    pub fn execute<F, R>(
        &self,
        f: F,
    ) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.thread_pool.execute(f)
    }
}

#[derive(Debug, Clone)]
pub struct WrappedUtils {
    utils: Arc<Mutex<Utils>>,
}

impl WrappedUtils {
    #[must_use]
    pub fn new(utils: Utils) -> Self {
        Self { utils: Arc::new(Mutex::new(utils)) }
    }

    pub fn allocate(
        &mut self,
        allocatable: &WrappedLayer,
    ) -> bool {
        safe_lock(&self.utils).allocate(allocatable)
    }

    pub fn deallocate(
        &mut self,
        allocatable: &WrappedLayer,
    ) {
        safe_lock(&self.utils).deallocate(allocatable);
    }

    pub fn allocate_trainable(
        &mut self,
        allocatable: &WrappedTrainableLayer,
    ) -> bool {
        safe_lock(&self.utils).allocate_trainable(allocatable)
    }

    pub fn deallocate_trainable(
        &mut self,
        allocatable: &WrappedTrainableLayer,
    ) {
        safe_lock(&self.utils).deallocate_trainable(allocatable);
    }

    #[must_use]
    pub fn get_max_allocated_size(&self) -> usize {
        safe_lock(&self.utils).get_max_allocated_size()
    }

    #[must_use]
    pub fn get_multi_progress(&self) -> Arc<MultiProgress> {
        safe_lock(&self.utils).get_multi_progress()
    }

    pub fn execute<F, R>(
        &self,
        f: F,
    ) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        safe_lock(&self.utils).execute(f)
    }
}
