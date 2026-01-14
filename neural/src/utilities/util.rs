use utils::safer::safe_lock;

use std::sync::{Arc, Mutex};

use crate::layer::layer_trait::{WrappedLayer, WrappedTrainableLayer};
use alloc::alloc_manager::{AllocManager, WrappedAllocManager};

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
    mutli_progress: Arc<MultiProgress>,
    thread_pool: WrappedThreadPool,
    test_mode: bool,
    workspace: String,
}

impl Utils {
    #[must_use]
    pub fn new(
        cpu_memory: usize,
        num_threads: usize,
    ) -> Self {
        Self {
            mutli_progress: Arc::new(MultiProgress::new()),
            thread_pool: WrappedThreadPool::new(num_threads),
            test_mode: false,
            workspace: String::new(),
        }
    }

    #[must_use]
    pub fn new_with_test_mode(
        cpu_memory: usize,
        num_threads: usize,
        workspace: String,
    ) -> Self {
        Self {
            mutli_progress: Arc::new(MultiProgress::new()),
            thread_pool: WrappedThreadPool::new(num_threads),
            test_mode: true,
            workspace,
        }
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

    #[must_use]
    pub const fn is_test_mode(&self) -> bool {
        self.test_mode
    }

    #[must_use]
    pub fn get_workspace(&self) -> &str {
        &self.workspace
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

    #[must_use]
    pub fn is_test_mode(&self) -> bool {
        safe_lock(&self.utils).is_test_mode()
    }

    #[must_use]
    pub fn get_workspace(&self) -> String {
        safe_lock(&self.utils).get_workspace().to_string()
    }
}
