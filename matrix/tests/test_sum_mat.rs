use matrix::mat::WrappedMatrix;
use matrix::sum_mat::MyInteger;
use matrix::sum_mat::SumMatrix;

#[test]
fn test_integration_sum_matrices() {
    let matrix = WrappedMatrix::new(3, 3);
    let mut sum_matrix = SumMatrix::new(matrix);

    sum_matrix.set_val(0, 0, 1).unwrap();

    sum_matrix.set_val_unchecked(0, 1, 2);

    assert_eq!(sum_matrix.get_row_sum(0), MyInteger(3));
    assert_eq!(sum_matrix.get_val(0, 0).unwrap(), MyInteger(1));
    assert_eq!(sum_matrix.get_val(0, 1).unwrap(), MyInteger(2));
    // assert get_ratio
    let epsilon = 1e-6;
    assert!((sum_matrix.get_ratio(0, 0).unwrap() - 1.0 / 3.0).abs() < epsilon);
    assert!((sum_matrix.get_ratio(0, 1).unwrap() - 2.0 / 3.0).abs() < epsilon);
}
