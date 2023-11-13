use crate::common::{HexagonalError, HexagonalResult};
use crate::game::{Game, GameResult, Player};
use crate::hexboard::{Coordinate, Direction, DirectionIterator, RoundHexBoard};
use std::cmp::Ordering;

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum TumbleweedPiece {
    Black,
    White,
    Neutral,
}

impl TumbleweedPiece {
    pub fn opponent(&self) -> TumbleweedPiece {
        match self {
            TumbleweedPiece::Black => TumbleweedPiece::White,
            TumbleweedPiece::White => TumbleweedPiece::Black,
            _ => *self,
        }
    }
}

impl From<Player> for TumbleweedPiece {
    fn from(val: Player) -> Self {
        match val {
            Player::Black => TumbleweedPiece::Black,
            Player::White => TumbleweedPiece::White,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TumbleweedField {
    pub stack: u8,
    pub color: Option<TumbleweedPiece>,
    pub los: [u8; 2],
}

#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub enum TumbleweedMove {
    Setup(Coordinate, Coordinate),
    Swap,
    Play(Coordinate),
    Pass,
}

#[derive(Debug, Clone)]
pub struct Tumbleweed {
    consecutive_passes: u8,
    pub current: Player,
    pub valid_moves: Vec<TumbleweedMove>,
    played_moves: Vec<TumbleweedMove>,
    board: RoundHexBoard<TumbleweedField>,
}

impl Game for Tumbleweed {
    type Move = TumbleweedMove;

    fn game_over(&self) -> bool {
        self.consecutive_passes >= 2 || self.valid_moves.is_empty()
    }

    fn result(&self) -> GameResult {
        let (b, w) = self.score();
        match b.cmp(&w) {
            Ordering::Less => GameResult::WhiteWin,
            Ordering::Equal => GameResult::Draw,
            Ordering::Greater => GameResult::BlackWin,
        }
    }

    fn valid_moves(&self) -> &[TumbleweedMove] {
        self.valid_moves.as_slice()
    }

    fn play(&mut self, tmove: TumbleweedMove) -> HexagonalResult<()> {
        if !self.valid_moves.iter().any(|&x| x == tmove) {
            return Err(HexagonalError::new(format!("Invalid move: {:?}", tmove)));
        }
        self.played_moves.push(tmove);

        match tmove {
            TumbleweedMove::Setup(b, w) => self.setup((0, 0).into(), b, w),
            TumbleweedMove::Swap => self.swap_colors(),
            TumbleweedMove::Play(m) => {
                let stack = self.board[m].los[self.current as usize];
                self.place(self.current.into(), stack, m);
            }
            TumbleweedMove::Pass => {
                self.consecutive_passes += 1;
            }
        };

        if tmove != TumbleweedMove::Pass {
            self.consecutive_passes = 0;
        }

        self.current = self.next_player();
        self.update_valids();
        Ok(())
    }

    fn current_player(&self) -> Player {
        self.current
    }

    fn next_player(&self) -> Player {
        self.current.opponent()
    }

    fn last_move(&self) -> Option<TumbleweedMove> {
        self.played_moves.last().copied()
    }
}

impl Tumbleweed {
    pub fn new(size: u8) -> Tumbleweed {
        let mut t = Tumbleweed {
            consecutive_passes: 0,
            board: RoundHexBoard::new(size),
            current: Player::Black,
            valid_moves: vec![],
            played_moves: vec![],
        };
        t.update_valids();
        t
    }

    fn swap_colors(&mut self) {
        for f in self.board.iter_fields_mut() {
            f.color = f.color.map(|c| c.opponent());
            f.los.reverse();
        }
    }

    fn update_valids(&mut self) {
        if self.consecutive_passes >= 2 {
            self.valid_moves = vec![];
            return;
        }

        self.valid_moves = match self.last_move() {
            Some(m) => {
                if let TumbleweedMove::Setup(_, _) = m {
                    let mut valids = self.gen_valids();
                    valids.push(TumbleweedMove::Swap);
                    valids
                } else {
                    let mut valids = self.gen_valids();
                    valids.push(TumbleweedMove::Pass);
                    valids
                }
            }
            None => self.gen_start_moves(),
        };
    }

    fn gen_valids(&self) -> Vec<TumbleweedMove> {
        self.board
            .iter_coord_fields()
            .filter(|(_, field)| {
                let lc = field.los[self.current as usize];
                let lo = field.los[self.current.opponent() as usize];
                (lc > field.stack)
                    && (lc >= lo)
                    && (lc < 6)
                    && !(field.color == Some(self.current.into()) && (lc > lo + 1))
            })
            .map(|(coord, _)| TumbleweedMove::Play(coord))
            .collect()
    }

    fn gen_start_moves(&self) -> Vec<TumbleweedMove> {
        let mut ret = Vec::with_capacity(
            (self.board.as_slice().len() - 1) * (self.board.as_slice().len() - 2),
        );
        for b in self.board.iter_coords() {
            if *b == (0, 0).into() {
                continue;
            }
            for w in self.board.iter_coords() {
                if *w == (0, 0).into() || b == w {
                    continue;
                }
                ret.push(TumbleweedMove::Setup(*b, *w));
            }
        }
        ret
    }

    pub fn place(&mut self, color: TumbleweedPiece, stack: u8, coord: Coordinate) {
        let cur = &self.board[coord];
        let prev_color = cur.color;
        self.board[coord] = TumbleweedField {
            stack,
            color: Some(color),
            los: cur.los,
        };
        update_los(&mut self.board, color, prev_color, coord);
    }

    pub fn score(&self) -> (i16, i16) {
        let pointsb = self
            .board
            .iter_fields()
            .filter(|field| {
                let lb = field.los[TumbleweedPiece::Black as usize];
                let lw = field.los[TumbleweedPiece::White as usize];
                ((lb > lw) && (lb > field.stack))
                    || ((field.color == Some(TumbleweedPiece::Black)) && (field.stack >= lw))
            })
            .count() as i16;

        let pointsw = self
            .board
            .iter_fields()
            .filter(|field| {
                let lb = field.los[TumbleweedPiece::Black as usize];
                let lw = field.los[TumbleweedPiece::White as usize];
                ((lw > lb) && (lw > field.stack))
                    || ((field.color == Some(TumbleweedPiece::White)) && (field.stack >= lb))
            })
            .count() as i16;

        (pointsb, pointsw)
    }

    fn setup(&mut self, neutral: Coordinate, black: Coordinate, white: Coordinate) {
        self.place(TumbleweedPiece::Neutral, 2, neutral);
        self.place(TumbleweedPiece::Black, 1, black);
        self.place(TumbleweedPiece::White, 1, white);
    }

    pub fn board(&self) -> &RoundHexBoard<TumbleweedField> {
        &self.board
    }
}

fn update_los(
    board: &mut RoundHexBoard<TumbleweedField>,
    color: TumbleweedPiece,
    prev_color: Option<TumbleweedPiece>,
    coord: Coordinate,
) {
    if prev_color == Some(color) || color == TumbleweedPiece::Neutral {
        return;
    }

    let oppo = color.opponent();

    for dir in [Direction::XY, Direction::YZ, Direction::ZX].iter() {
        let mut to_update_l = Vec::with_capacity(board.size() as usize * 2);
        let mut to_update_r = Vec::with_capacity(board.size() as usize * 2);

        let itup = DirectionIterator::new(coord, *dir);
        let itdown = DirectionIterator::new(coord, -*dir);

        let mut other_piece_l = None;
        for cc in itup.skip(1) {
            if !board.valid_coord(cc) {
                break;
            }
            to_update_l.push(cc);
            if board[cc].stack > 0 {
                other_piece_l = board[cc].color;
                break;
            }
        }

        let mut other_piece_r = None;
        for cc in itdown.skip(1) {
            if !board.valid_coord(cc) {
                break;
            }
            to_update_r.push(cc);
            if board[cc].stack > 0 {
                other_piece_r = board[cc].color;
                break;
            }
        }

        if (other_piece_l == Some(oppo) && prev_color.is_none()) || (prev_color == Some(oppo)) {
            for c in to_update_r {
                board[c].los[oppo as usize] -= 1;
                board[c].los[color as usize] += 1;
            }
        } else if (other_piece_l.is_none())
            || (other_piece_l == Some(TumbleweedPiece::Neutral))
            || (prev_color == Some(TumbleweedPiece::Neutral))
        {
            for c in to_update_r {
                board[c].los[color as usize] += 1;
            }
        }

        if (other_piece_r == Some(oppo) && prev_color.is_none()) || (prev_color == Some(oppo)) {
            for c in to_update_l {
                board[c].los[oppo as usize] -= 1;
                board[c].los[color as usize] += 1;
            }
        } else if (other_piece_r.is_none())
            || (other_piece_r == Some(TumbleweedPiece::Neutral))
            || (prev_color == Some(TumbleweedPiece::Neutral))
        {
            for c in to_update_l {
                board[c].los[color as usize] += 1;
            }
        }
    }
}
