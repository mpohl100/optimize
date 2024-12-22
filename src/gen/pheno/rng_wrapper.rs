use crate::evol::rng::RandomNumberGenerator;

use std::collections::VecDeque;

pub trait RngWrapper{
    fn fetch_uniform(&mut self, min: f32, max: f32, count: usize) -> VecDeque<f32>;
}

pub struct RealRng<'a> {
    rng: &'a mut RandomNumberGenerator,
}

impl<'a> RealRng<'a> {
    pub fn new(rng: &'a mut RandomNumberGenerator) -> Self {
        RealRng { rng }
    }
}

impl<'a> RngWrapper for RealRng<'a> {
    fn fetch_uniform(&mut self, min: f32, max: f32, count: usize) -> VecDeque<f32> {
        self.rng.fetch_uniform(min, max, count)
    }
}

pub struct FakeRng {
    values: VecDeque<f32>,
}

impl FakeRng {
    pub fn new(values: Vec<f32>) -> Self {
        FakeRng {
            values: values.into_iter().collect(),
        }
    }
}

impl RngWrapper for FakeRng {
    fn fetch_uniform(&mut self, min: f32, max: f32, count: usize) -> VecDeque<f32> {
        let mut result = VecDeque::new();
        for _ in 0..count {

            let val = self.values.pop_front();
            match val {
                Some(val) => {
                    // Check
                    if val < min || val > max {
                        panic!("Value out of range");
                    }
                    result.push_back(val);
                },
                None => panic!("No more values"),
            };
        }

        result
    }
}
