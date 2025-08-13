pub trait Allocatable {
    fn allocate(&mut self);
    fn deallocate(&mut self);
    fn is_allocated(&self) -> bool;
    fn get_size(&self) -> usize;
    fn mark_for_use(&mut self);
    fn free_from_use(&mut self);
    fn is_in_use(&self) -> bool;
}

pub trait WrappedAllocatableTrait: Clone {
    fn allocate(&self);
    fn deallocate(&self);
    fn is_allocated(&self) -> bool;
    fn get_size(&self) -> usize;
    fn mark_for_use(&mut self);
    fn free_from_use(&mut self);
    fn is_in_use(&self) -> bool;
}
