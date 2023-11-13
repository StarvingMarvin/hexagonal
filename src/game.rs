use std::fmt::Debug;
use std::hash::Hash;

use crate::common::HexagonalResult;

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

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum GameResult {
    BlackWin,
    Draw,
    WhiteWin,
}

pub trait MoveTrait: PartialEq + Eq + Copy + Clone + Hash + Debug + Send + Sync {}
impl<T> MoveTrait for T where T: PartialEq + Eq + Copy + Clone + Hash + Debug + Send + Sync {}

pub trait Game: Clone + Send {
    type Move: MoveTrait;

    fn game_over(&self) -> bool;
    fn result(&self) -> GameResult;
    fn valid_moves(&self) -> &[Self::Move];
    fn play(&mut self, action: Self::Move) -> HexagonalResult<()>;
    fn current_player(&self) -> Player;
    fn next_player(&self) -> Player;
    fn last_move(&self) -> Option<Self::Move>;
    fn is_valid(&self, mv: Self::Move) -> bool {
        self.valid_moves().contains(&mv)
    }
}
