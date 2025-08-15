use std::iter::Sum;

use crate::mat::{OutOfRangeError, WrappedMatrix};

use num_traits::NumCast;
use utils::safer::safe_lock;

struct SumMatrix {
    matrix: WrappedMatrix<i64>,
    row_sum: Vec<i64>,
}

impl SumMatrix {
    pub fn new(matrix: WrappedMatrix<i64>) -> Self {
        let row_sum = safe_lock(&matrix.mat).iter().map(|row| row.iter().sum()).collect::<Vec<_>>();
        Self { matrix, row_sum }
    }

    pub fn get_row_sum(
        &self,
        row: usize,
    ) -> i64 {
        self.row_sum[row]
    }

    pub fn set_val_unchecked(
        &mut self,
        row: usize,
        col: usize,
        val: i64,
    ) {
        let previous_val = self.matrix.get_unchecked(row, col);
        let delta = val - previous_val;
        self.matrix.set_mut_unchecked(row, col, val);
        self.row_sum[row] += delta;
    }

    pub fn set_val(
        &mut self,
        row: usize,
        col: usize,
        val: i64,
    ) -> Result<(), String> {
        if row >= self.matrix.rows() || col >= self.matrix.cols() {
            return Err("Index out of bounds".to_string());
        }
        let previous_val = self.matrix.get_unchecked(row, col);
        let delta = val - previous_val;
        self.set_val_unchecked(row, col, val);
        self.row_sum[row] += delta;
        Ok(())
    }

    pub fn get_val(
        &self,
        row: usize,
        col: usize,
    ) -> Result<i64, OutOfRangeError> {
        self.matrix.get(row, col)
    }

    pub fn get_val_unchecked(
        &self,
        row: usize,
        col: usize,
    ) -> i64 {
        self.matrix.get_unchecked(row, col)
    }

    pub fn get_ratio(
        &self,
        row: usize,
        col: usize,
    ) -> Result<f64, String> {
        if row >= self.matrix.rows() || col >= self.matrix.cols() {
            return Err("Index out of bounds".to_string());
        }
        let val = self.matrix.get_unchecked(row, col);
        let row_sum = self.row_sum[row];
        if row_sum == 0 {
            return Err("Division by zero".to_string());
        }
        let val_f64: f64 = NumCast::from(val).unwrap();
        let row_sum_f64: f64 = NumCast::from(row_sum).unwrap();
        Ok(val_f64 / row_sum_f64)
    }
}
