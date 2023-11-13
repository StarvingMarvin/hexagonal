use std::{error::Error, fmt::Display};

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum Player {
    Black,
    White,
}

impl Player {
    pub fn opponent(&self) -> Self {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
    }
}

impl From<Player> for usize {
    fn from(val: Player) -> Self {
        val as usize
    }
}

#[derive(Debug)]
pub struct HexagonalError {
    message: String,
}

impl Display for HexagonalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for HexagonalError {

}

impl HexagonalError {
    pub fn new(message: String) -> Self {
        HexagonalError { message }
    }
}

pub type HexagonalResult<T> = Result<T, HexagonalError>;
