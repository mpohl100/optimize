use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use matrix::mat::WrappedMatrix;
use matrix::sum_mat::SumMatrix;

fn benchmark_matrix_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_creation");

    for size in [10, 50, 100, 200, 500].iter() {
        group.bench_with_input(BenchmarkId::new("WrappedMatrix", size), size, |b, &size| {
            b.iter(|| {
                let _matrix = black_box(WrappedMatrix::<f64>::new(size, size));
            });
        });
    }

    group.finish();
}

fn benchmark_matrix_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_access");

    for size in [50, 100, 200].iter() {
        let matrix = WrappedMatrix::<f64>::new(*size, *size);

        // Initialize matrix with test data
        for i in 0..*size {
            for j in 0..*size {
                matrix.set_mut_unchecked(i, j, (i * j) as f64);
            }
        }

        group.bench_with_input(BenchmarkId::new("get_unchecked", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    for j in 0..size {
                        black_box(matrix.get_unchecked(i, j));
                    }
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("set_mut_unchecked", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    for j in 0..size {
                        matrix.set_mut_unchecked(i, j, black_box((i + j) as f64));
                    }
                }
            });
        });
    }

    group.finish();
}

fn benchmark_sum_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_matrix_operations");

    for size in [50, 100, 200].iter() {
        let base_matrix = WrappedMatrix::<i64>::new(*size, *size);
        let mut sum_matrix = SumMatrix::new(base_matrix);

        // Initialize with test data
        for i in 0..*size {
            for j in 0..*size {
                sum_matrix.set_val(i, j, (i + j + 1) as i64).unwrap();
            }
        }

        group.bench_with_input(BenchmarkId::new("get_row_sum", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(sum_matrix.get_row_sum(i));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("get_ratio", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    for j in 0..size {
                        black_box(sum_matrix.get_ratio(i, j).unwrap());
                    }
                }
            });
        });

        group.bench_with_input(
            BenchmarkId::new("set_val_with_sum_update", size),
            size,
            |b, &size| {
                b.iter(|| {
                    for i in 0..size {
                        for j in 0..size {
                            sum_matrix.set_val(i, j, black_box((i * j + 1) as i64)).unwrap();
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_matrix_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_iteration");

    for size in [100, 200, 500].iter() {
        let matrix = WrappedMatrix::<f64>::new(*size, *size);

        // Initialize with test data
        for i in 0..*size {
            for j in 0..*size {
                matrix.set_mut_unchecked(i, j, (i * j) as f64);
            }
        }

        group.bench_with_input(BenchmarkId::new("sequential_access", size), size, |b, &size| {
            b.iter(|| {
                let mut sum = 0.0;
                for i in 0..size {
                    for j in 0..size {
                        sum += matrix.get_unchecked(i, j);
                    }
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("random_access", size), size, |b, &size| {
            b.iter(|| {
                let mut sum = 0.0;
                for k in 0..(size * size) {
                    let i = k % size;
                    let j = (k * 17) % size; // Pseudo-random access pattern
                    sum += matrix.get_unchecked(i, j);
                }
                black_box(sum);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_matrix_creation,
    benchmark_matrix_access,
    benchmark_sum_matrix_operations,
    benchmark_matrix_iteration
);
criterion_main!(benches);
