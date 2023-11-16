use crate::common::{HexagonalError, HexagonalResult};
use crate::game::{Game, GameResult, Player};
use crate::hexboard::{BoardCoord, Direction, DirectionIterator, RoundHexBoard};
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
}

#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub enum TumbleweedMove {
    Setup(BoardCoord, BoardCoord),
    Swap,
    Play(BoardCoord),
    Pass,
}

struct GenStartMoves {
    coords: &'static [BoardCoord],
    b: usize,
    w: usize,
    zero: usize,
    cnt: usize,
}

impl GenStartMoves {
    fn new(coords: &'static [BoardCoord]) -> Self {
        GenStartMoves {
            coords,
            b: 0,
            w: 1,
            zero: coords.len() / 2,
            cnt: 0,
        }
    }
}

impl Iterator for GenStartMoves {
    type Item = TumbleweedMove;

    fn next(&mut self) -> Option<Self::Item> {
        (self.b < self.coords.len()).then(|| {
            let ret = TumbleweedMove::Setup(self.coords[self.b], self.coords[self.w]);
            self.w += 1;
            while self.w == self.b || self.w == self.zero {
                self.w += 1;
            }
            if self.w >= self.coords.len() {
                self.w = 0;
                self.b += 1;
                if self.b == self.zero {
                    self.b += 1;
                }
            }
            self.cnt += 1;
            ret
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.coords.len();
        let total = (l - 1) * (l - 2);
        (total - self.cnt, Some(total - self.cnt))
    }
}

impl ExactSizeIterator for GenStartMoves {}

#[derive(Debug, Clone)]
pub struct Tumbleweed {
    consecutive_passes: u8,
    pub current: Player,
    pub valid_moves: Vec<TumbleweedMove>,
    played_moves: Vec<TumbleweedMove>,
    board: RoundHexBoard<TumbleweedField>,
    los: RoundHexBoard<[u8; 2]>,
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
        self.play_unchecked(tmove);
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
            los: RoundHexBoard::new(size),
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
        }
        for f in self.los.iter_fields_mut() {
            f.reverse();
        }
    }

    pub fn play_unchecked(&mut self, tmove: TumbleweedMove) {
        self.played_moves.push(tmove);

        match tmove {
            TumbleweedMove::Setup(b, w) => self.setup((0, 0).into(), b, w),
            TumbleweedMove::Swap => self.swap_colors(),
            TumbleweedMove::Play(m) => {
                let stack = self.los[m][self.current as usize];
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
    }

    fn update_valids(&mut self) {
        if self.consecutive_passes >= 2 {
            self.valid_moves = vec![];
            return;
        }

        match self.last_move() {
            Some(m) => {
                if let TumbleweedMove::Setup(_, _) = m {
                    self.gen_valids();
                    self.valid_moves.push(TumbleweedMove::Swap);
                    self.valid_moves.shrink_to(self.valid_moves.len() * 3 / 2);
                } else {
                    self.gen_valids();
                    self.valid_moves.push(TumbleweedMove::Pass);
                }
            }
            None => self.gen_start_moves(),
        };
    }

    fn gen_valids(&mut self) {
        self.valid_moves.clear();
        self.valid_moves.extend(
            self.board
                .iter_coords()
                .zip(self.board.iter_fields())
                .zip(self.los.iter_fields())
                .filter_map(|((coord, field), los)| {
                    let lc = los[self.current as usize];
                    let lo = los[self.current.opponent() as usize];
                    ((lc > field.stack)
                        && (lc >= lo)
                        && (lc < 6)
                        && !(field.color == Some(self.current.into()) && (lc > lo + 1)))
                        .then_some(TumbleweedMove::Play(*coord))
                }),
        );
    }

    fn gen_start_moves(&mut self) {
        self.valid_moves = GenStartMoves::new(self.board.get_coords()).collect();
    }

    pub fn place(&mut self, color: TumbleweedPiece, stack: u8, coord: BoardCoord) {
        let cur = &self.board[coord];
        let prev_color = cur.color;
        self.board[coord] = TumbleweedField {
            stack,
            color: Some(color),
        };
        update_los(&self.board, &mut self.los, color, prev_color, coord);
    }

    pub fn score(&self) -> (i16, i16) {
        let pointsb = self
            .los
            .iter_fields()
            .zip(self.board.iter_fields())
            .filter(|(los, field)| {
                let lb = los[TumbleweedPiece::Black as usize];
                let lw = los[TumbleweedPiece::White as usize];
                ((lb > lw) && (lb > field.stack))
                    || ((field.color == Some(TumbleweedPiece::Black)) && (field.stack >= lw))
            })
            .count() as i16;

        let pointsw = self
            .los
            .iter_fields()
            .zip(self.board.iter_fields())
            .filter(|(los, field)| {
                let lb = los[TumbleweedPiece::Black as usize];
                let lw = los[TumbleweedPiece::White as usize];
                ((lw > lb) && (lw > field.stack))
                    || ((field.color == Some(TumbleweedPiece::White)) && (field.stack >= lb))
            })
            .count() as i16;

        (pointsb, pointsw)
    }

    fn setup(&mut self, neutral: BoardCoord, black: BoardCoord, white: BoardCoord) {
        self.place(TumbleweedPiece::Neutral, 2, neutral);
        self.place(TumbleweedPiece::Black, 1, black);
        self.place(TumbleweedPiece::White, 1, white);
    }

    pub fn board(&self) -> &RoundHexBoard<TumbleweedField> {
        &self.board
    }

    pub fn played_moves(&self) -> &[TumbleweedMove] {
        &self.played_moves
    }
}

fn update_los(
    board: &RoundHexBoard<TumbleweedField>,
    los: &mut RoundHexBoard<[u8; 2]>,
    color: TumbleweedPiece,
    prev_color: Option<TumbleweedPiece>,
    coord: BoardCoord,
) {
    if prev_color == Some(color) || color == TumbleweedPiece::Neutral {
        return;
    }

    let oppo = color.opponent();
    let bs1 = board.size() as i8 - 1;

    for dir in [Direction::XY, Direction::YZ, Direction::ZX].iter() {
        let mut to_update_l = Vec::with_capacity(board.size() as usize * 2);
        let mut to_update_r = Vec::with_capacity(board.size() as usize * 2);

        let itup = DirectionIterator::new(coord, *dir);
        let itdown = DirectionIterator::new(coord, -*dir);

        // let mut cc = coord;
        let (cu, cd) = match dir {
            Direction::YZ => (coord.y, coord.z()),
            Direction::XY => (coord.x, coord.y),
            Direction::ZX => (coord.z(), coord.x),
            _ => panic!(),
        };

        let mut other_piece_l = None;
        for cc in itup.skip(1).take((bs1 - cu).min(bs1 + cd) as usize) {
            to_update_l.push(cc);
            if board[cc].stack > 0 {
                other_piece_l = board[cc].color;
                break;
            }
        }

        let mut other_piece_r = None;
        for cc in itdown.skip(1).take((bs1 + cu).min(bs1 - cd) as usize) {
            to_update_r.push(cc);
            if board[cc].stack > 0 {
                other_piece_r = board[cc].color;
                break;
            }
        }

        if (other_piece_l == Some(oppo) && prev_color.is_none()) || (prev_color == Some(oppo)) {
            for c in to_update_r {
                los[c][oppo as usize] -= 1;
                los[c][color as usize] += 1;
            }
        } else if (other_piece_l.is_none())
            || (other_piece_l == Some(TumbleweedPiece::Neutral))
            || (prev_color == Some(TumbleweedPiece::Neutral))
        {
            for c in to_update_r {
                los[c][color as usize] += 1;
            }
        }

        if (other_piece_r == Some(oppo) && prev_color.is_none()) || (prev_color == Some(oppo)) {
            for c in to_update_l {
                los[c][oppo as usize] -= 1;
                los[c][color as usize] += 1;
            }
        } else if (other_piece_r.is_none())
            || (other_piece_r == Some(TumbleweedPiece::Neutral))
            || (prev_color == Some(TumbleweedPiece::Neutral))
        {
            for c in to_update_l {
                los[c][color as usize] += 1;
            }
        }
    }
}
