//! Abstract interface describing a two player game.

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

use crate::common::HexagonalResult;


#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum Player {
    Black,
    White,
}

impl Player {

    #[inline]
    pub fn opponent(&self) -> Self {
        match self {
            Player::Black => Player::White,
            Player::White => Player::Black,
        }
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

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, Default)]
pub struct BW<T>([T; 2]);

impl<T> Index<Player> for BW<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: Player) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl<T> IndexMut<Player> for BW<T> {

    #[inline]
    fn index_mut(&mut self, index: Player) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl<T> BW<T> {

    #[inline]
    pub fn swap(&mut self) {
        self.0.reverse();
    }

    #[inline]
    pub fn black(&self) -> &T {
        &self.0[0]
    }

    #[inline]
    pub fn white(&self) -> &T {
        &self.0[1]
    }

    #[inline]
    pub fn take_black(self) -> T {
        let [b, _] = self.0;
        b
    }

    #[inline]
    pub fn take_white(self) -> T {
        let [_, w] = self.0;
        w
    }

}

impl<T> From<[T; 2]> for BW<T> {

    #[inline]
    fn from(value: [T; 2]) -> Self {
        BW(value)
    }
}

impl<T> From<(T, T)> for BW<T> {

    #[inline]
    fn from(value: (T, T)) -> Self {
        BW([value.0, value.1])
    }
}
