use matrix::mat::WrappedMatrix;
use matrix::sum_mat::MyInteger;
use matrix::sum_mat::SumMatrix;
use proptest::prelude::*;
// Property-based tests for WrappedMatrix
proptest! {
    #[test]
    fn test_matrix_dimensions_property(
        rows in 1..100usize,
        cols in 1..100usize,
    ) {
        let matrix = WrappedMatrix::<f64>::new(rows, cols);
        assert_eq!(matrix.rows(), rows);
        assert_eq!(matrix.cols(), cols);
    }

    #[test]
    fn test_matrix_set_get_property(
        rows in 1..20usize,
        cols in 1..20usize,
        value in -1000.0..1000.0f64,
    ) {
        let matrix = WrappedMatrix::<f64>::new(rows, cols);
        let row = rows / 2;
        let col = cols / 2;

        matrix.set_mut_unchecked(row, col, value);
        let retrieved = matrix.get_unchecked(row, col);

        assert!((retrieved - value).abs() < f64::EPSILON);
    }

    #[test]
    fn test_matrix_bounds_checking(
        rows in 1..10usize,
        cols in 1..10usize,
        test_row in 0..20usize,
        test_col in 0..20usize,
    ) {
        let matrix = WrappedMatrix::<f64>::new(rows, cols);

        if test_row >= rows || test_col >= cols {
            assert!(matrix.get(test_row, test_col).is_err());
        } else {
            assert!(matrix.get(test_row, test_col).is_ok());
        }
    }
}

// Property-based tests for SumMatrix
proptest! {
    #[test]
    fn test_sum_matrix_row_sum_property(
        rows in 1..10usize,
        cols in 1..10usize,
        values in prop::collection::vec(-100..100i64, 1..10),
    ) {
        let base_matrix = WrappedMatrix::<MyInteger>::new(rows, cols);
        let mut sum_matrix = SumMatrix::new(base_matrix);

        let row = rows / 2;
        let mut expected_sum = 0i64;

        for (i, &value) in values.iter().enumerate() {
            if i < cols {
                sum_matrix.set_val(row, i, value).unwrap();
                expected_sum += value;
            }
        }

        let actual_sum = sum_matrix.get_row_sum(row);
        assert_eq!(actual_sum, MyInteger(expected_sum));
    }

    #[test]
    fn test_sum_matrix_ratio_property(
        rows in 1..10usize,
        cols in 2..10usize,
        value1 in 1..100i64,
        value2 in 1..100i64,
    ) {
        let base_matrix = WrappedMatrix::<MyInteger>::new(rows, cols);
        let mut sum_matrix = SumMatrix::new(base_matrix);

        let row = 0;
        sum_matrix.set_val(row, 0, value1).unwrap();
        sum_matrix.set_val(row, 1, value2).unwrap();

        let ratio1 = sum_matrix.get_ratio(row, 0).unwrap();
        let ratio2 = sum_matrix.get_ratio(row, 1).unwrap();

        // Ratios should sum to 1.0 (approximately)
        let total_ratio = ratio1 + ratio2;
        assert!((total_ratio - 1.0).abs() < 1e-10);

        // Individual ratios should be correct
        let total = value1 + value2;
        let expected_ratio1 = value1 as f64 / total as f64;
        let expected_ratio2 = value2 as f64 / total as f64;

        assert!((ratio1 - expected_ratio1).abs() < 1e-10);
        assert!((ratio2 - expected_ratio2).abs() < 1e-10);
    }
}
