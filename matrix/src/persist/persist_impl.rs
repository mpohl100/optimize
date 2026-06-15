use crate::mat::WrappedMatrix;
use crate::persist::traits::PersistableValue;
use fs2::FileExt;
use std::error::Error;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;

pub fn save<T: PersistableValue + From<f64>>(
    path: String,
    matrix: &WrappedMatrix<T>,
) -> Result<(), Box<dyn Error>> {
    // Ensure the directory exists
    let p = Path::new(&path);
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).expect("Failed to create directory");
    }
    // create a lock file which acts as a lock
    let lock_file_path = format!("{path}.lock");
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;
    // Save weights and biases to a file at the specified path
    let mut file = File::create(path)?;
    writeln!(file, "{} {}", matrix.rows(), matrix.cols())?;
    for i in 0..matrix.rows() {
        for j in 0..matrix.cols() {
            write!(file, "{};", matrix.get_unchecked(i, j).to_string_for_matrix())?;
        }
        writeln!(file)?;
    }
    writeln!(file)?;
    Ok(())
}

pub fn read<T: PersistableValue + From<f64>>(
    path: String
) -> Result<WrappedMatrix<T>, Box<dyn Error>> {
    // create a lock file which acts as a lock
    let lock_file_path = format!("{path}.lock");
    let lock_file = File::create(&lock_file_path)?;
    lock_file.lock_exclusive()?;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut weights = WrappedMatrix::new(1, 1);
    if let Some(Ok(line)) = lines.next() {
        let mut parts = line.split_whitespace();
        let rows = parts.next().unwrap().parse::<usize>()?;
        let cols = parts.next().unwrap().parse::<usize>()?;
        weights = WrappedMatrix::new(rows, cols);
        for i in 0..rows {
            if let Some(Ok(line)) = lines.next() {
                let parts = line.split(';').collect::<Vec<_>>();
                // parts len must be euqal to cols
                if parts.len() - 1 != cols {
                    return Err(format!(
                        "Invalid weight format cause of cols: expected {}, found {}",
                        cols,
                        parts.len() - 1
                    )
                    .into());
                }
                for j in 0..cols {
                    let part = parts.get(j);
                    if let Some(p) = part {
                        let figures = p.split_whitespace().collect::<Vec<_>>();
                        if figures.len() == 1 || figures.len() == 4 {
                            let val = T::from_string_for_matrix(figures[0])?;
                            weights.set_mut_unchecked(i, j, val);
                        } else {
                            return Err("Invalid weight format".into());
                        }
                    }
                }
            }
        }
    }
    Ok(weights)
}
