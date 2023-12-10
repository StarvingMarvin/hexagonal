use std::{error::Error, fmt::Display};

#[derive(Debug)]
pub struct HexagonalError {
    message: String,
}

impl Display for HexagonalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for HexagonalError {}

impl HexagonalError {
    pub fn new(message: String) -> Self {
        HexagonalError { message }
    }
}

pub type HexagonalResult<T> = Result<T, HexagonalError>;
