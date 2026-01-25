use crate::persistable_matrix::PersistableValue;
use std::error::Error;

#[derive(Default, Debug, Clone, Copy)]
pub struct Weight {
    pub value: f64,
    pub grad: f64,
    pub m: f64,
    pub v: f64,
}

impl From<f64> for Weight {
    fn from(value: f64) -> Self {
        Self { value, grad: 0.0, m: 0.0, v: 0.0 }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Bias {
    pub value: f64,
    pub grad: f64,
    pub m: f64,
    pub v: f64,
}

impl From<f64> for Bias {
    fn from(value: f64) -> Self {
        Self { value, grad: 0.0, m: 0.0, v: 0.0 }
    }
}

#[derive(Debug, Default, Clone)]
pub struct WeightEntry(pub Weight);

impl From<f64> for WeightEntry {
    fn from(value: f64) -> Self {
        Self(Weight { value, grad: 0.0, m: 0.0, v: 0.0 })
    }
}

impl PersistableValue for WeightEntry {
    fn to_string_for_matrix(&self) -> String {
        format!("{} {} {} {}", self.0.value, self.0.grad, self.0.m, self.0.v)
    }

    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 4 {
            return Err("Invalid weight entry format".into());
        }
        let value = parts[0].parse::<f64>()?;
        let grad = parts[1].parse::<f64>()?;
        let m = parts[2].parse::<f64>()?;
        let v = parts[3].parse::<f64>()?;
        Ok(Self(Weight { value, grad, m, v }))
    }
}

#[derive(Debug, Default, Clone)]
pub struct BiasEntry(pub Bias);

impl From<f64> for BiasEntry {
    fn from(value: f64) -> Self {
        Self(Bias { value, grad: 0.0, m: 0.0, v: 0.0 })
    }
}

impl PersistableValue for BiasEntry {
    fn to_string_for_matrix(&self) -> String {
        format!("{} {} {} {}", self.0.value, self.0.grad, self.0.m, self.0.v)
    }

    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() != 4 {
            return Err("Invalid bias entry format".into());
        }
        let value = parts[0].parse::<f64>()?;
        let grad = parts[1].parse::<f64>()?;
        let m = parts[2].parse::<f64>()?;
        let v = parts[3].parse::<f64>()?;
        Ok(Self(Bias { value, grad, m, v }))
    }
}

#[derive(Debug, Default, Clone)]
pub struct NumberEntry(pub f64);

impl From<f64> for NumberEntry {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl PersistableValue for NumberEntry {
    fn to_string_for_matrix(&self) -> String {
        format!("{}", self.0)
    }

    fn from_string_for_matrix(s: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        let value = s.parse::<f64>()?;
        Ok(Self(value))
    }
}
