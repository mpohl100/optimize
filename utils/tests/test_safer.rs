use std::sync::{Arc, Mutex};
use std::thread;
use utils::safer::safe_lock;

#[test]
fn test_safe_lock_normal_operation() {
    let data = Mutex::new(42);
    let guard = safe_lock(&data);
    assert_eq!(*guard, 42);
}

#[test]
fn test_safe_lock_with_multiple_threads() {
    let data = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for i in 0..10 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let mut guard = safe_lock(&data);
            *guard += i;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let guard = safe_lock(&data);
    assert_eq!(*guard, (0..10).sum::<i32>());
}

#[test]
fn test_safe_lock_with_poison_recovery() {
    let data = Arc::new(Mutex::new(Vec::new()));
    let data_clone = Arc::clone(&data);

    // Create a thread that will panic while holding the lock
    let handle = thread::spawn(move || {
        let mut guard = safe_lock(&data_clone);
        guard.push(1);
        panic!("Intentional panic");
    });

    // Thread should panic
    assert!(handle.join().is_err());

    // But we should still be able to access the data
    let guard = safe_lock(&data);
    assert_eq!(*guard, vec![1]);
}