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

