use crate::common::{HexagonalError, HexagonalResult};
use crate::game::{Game, GameResult, Player};
use crate::hexboard::{BoardCoord, Direction, DirectionIterator, RoundHexBoard};
use std::cmp::Ordering;
use std::iter::once;
use std::cell::OnceCell;

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

pub struct GenStartMoves {
    coords: &'static [BoardCoord],
    b: usize,
    w: usize,
    zero: usize,
    cnt: usize,
}

impl GenStartMoves {
    pub fn new(coords: &'static [BoardCoord]) -> Self {
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

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.cnt += n;
        let l2 = self.coords.len() - 2;
        self.b = self.cnt / l2;
        if self.b >= self.zero {
            self.b += 1;
        }
        self.w = self.cnt % l2;
        if self.w >= self.b {
            self.w += 1;
        }
        if self.w >= self.zero {
            self.w += 1;
        }
        self.next()
    }
}

impl ExactSizeIterator for GenStartMoves {}

#[derive(Debug, Clone)]
pub struct Tumbleweed {
    consecutive_passes: u8,
    pub current: Player,
    valid_moves: OnceCell<Vec<TumbleweedMove>>,
    played_moves: Vec<TumbleweedMove>,
    board: RoundHexBoard<TumbleweedField>,
    los: RoundHexBoard<[u8; 2]>,
}

impl Game for Tumbleweed {
    type Move = TumbleweedMove;

    fn game_over(&self) -> bool {
        self.consecutive_passes >= 2
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
        self.valid_moves.get_or_init(||
            GenStartMoves::new(self.board().get_coords()).collect()
        ).as_slice()
    }

    fn is_valid(&self, mv: Self::Move) -> bool {
        match self.played_moves.len() {
            0 => matches!(mv,
                     TumbleweedMove::Setup(b, w)
                     if b != (0, 0).into() &&
                         w != (0, 0).into() &&
                         b != w
            ),
            1 => match mv {
                TumbleweedMove::Swap => true,
                TumbleweedMove::Play(c) if self.los[c][1] % 2 > 0 => true,
                _ => false
            },
            _ => !self.game_over() && match mv {
                TumbleweedMove::Pass => true,
                TumbleweedMove::Play(c) if self.los[c][self.current as usize] % 2 > 0 => true,
                _ => false
            }
        }
    }

    fn play(&mut self, tmove: TumbleweedMove) -> HexagonalResult<()> {
        if !self.is_valid(tmove) {
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
        let t = Tumbleweed {
            consecutive_passes: 0,
            board: RoundHexBoard::new(size),
            los: RoundHexBoard::new(size),
            current: Player::Black,
            valid_moves: OnceCell::new(),
            played_moves: vec![],
        };
        // TODO: what to do if someone insists on full list of `Setup` moves?
        // t.update_valids();
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
                let stack = self.los[m][self.current as usize] / 2;
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
        let last = &self.last_move();
        let los = &self.los;
        let coords = &self.board.get_coords();
        self.valid_moves.get_or_init(|| vec![]);
        let valid = &mut self.valid_moves.get_mut().unwrap();
        valid.clear();
        if self.consecutive_passes >= 2 {
            return;
        }

        match last {
            Some(m) => {
                valid.extend(gen_valids(los, self.current as usize));
                if let TumbleweedMove::Setup(_, _) = m {
                    valid.push(TumbleweedMove::Swap);
                } else {
                    valid.push(TumbleweedMove::Pass);
                }
            }
            None => valid.extend(GenStartMoves::new(coords)),
        };
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

pub fn gen_valids(los: &RoundHexBoard<[u8; 2]>, color: usize) -> impl Iterator<Item=TumbleweedMove> + '_ {
    los.iter_coord_fields()
        .filter_map(move |(coord, &los)| {
            ((los[color] % 2) > 0).then_some(TumbleweedMove::Play(coord))
        })
}

fn update_los(
    board: &RoundHexBoard<TumbleweedField>,
    los: &mut RoundHexBoard<[u8; 2]>,
    color: TumbleweedPiece,
    prev_color: Option<TumbleweedPiece>,
    coord: BoardCoord,
) {
    if color == TumbleweedPiece::Neutral {
        return;
    }

    let oppo = color.opponent();

    if prev_color == Some(color) {
        los[coord][color as usize] &= 254;
        los[coord][oppo as usize] &= 254;
        return;
    }

    let bs1 = board.size() as i8 - 1;


    for dir in [Direction::XY, Direction::YZ, Direction::ZX].iter() {
        let mut to_update_l = Vec::with_capacity(board.size() as usize * 2);
        let mut to_update_r = Vec::with_capacity(board.size() as usize * 2);

        let itup = DirectionIterator::new(coord, *dir);
        let itdown = DirectionIterator::new(coord, -*dir);

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
            for &c in &to_update_r {
                los[c][oppo as usize] -= 2;
                los[c][color as usize] += 2;
            }
        } else if (other_piece_l.is_none())
            || (other_piece_l == Some(TumbleweedPiece::Neutral))
            || (prev_color == Some(TumbleweedPiece::Neutral))
        {
            for &c in &to_update_r {
                los[c][color as usize] += 2;
            }
        }

        if (other_piece_r == Some(oppo) && prev_color.is_none()) || (prev_color == Some(oppo)) {
            for &c in &to_update_l {
                los[c][oppo as usize] -= 2;
                los[c][color as usize] += 2;
            }
        } else if (other_piece_r.is_none())
            || (other_piece_r == Some(TumbleweedPiece::Neutral))
            || (prev_color == Some(TumbleweedPiece::Neutral))
        {
            for &c in &to_update_l {
                los[c][color as usize] += 2;
            }
        }

        for &c in once(&coord).chain(to_update_r.iter()).chain(to_update_l.iter()) {
            let lc = los[c][color as usize] / 2;
            let lo = los[c][oppo as usize] / 2;
            let field = &board[c];
            let validc = ((lc > field.stack)
                        && (lc >= lo)
                        && (lc < 6)
                        && !(field.color == Some(color) && (lc > lo + 1))) as u8;
            los[c][color as usize] &= 254;
            los[c][color as usize] |= validc;

            let valido = ((lo > field.stack)
                        && (lo >= lc)
                        && (lo < 6)
                        && !(field.color == Some(oppo) && (lo > lc + 1))) as u8;

            los[c][oppo as usize] &= 254;
            los[c][oppo as usize] |= valido;
        }
    }
}


#[cfg(test)]
mod tests {

    use super::*;
    use crate::hexboard::*;

    #[test]
    fn test_gen_setup() {
        let mut gen = GenStartMoves::new(get_coords(5));
        let mut gen2 = GenStartMoves::new(get_coords(5));
        for _ in 0..5 {
            let _ = gen2.next();
        }
        assert_eq!(gen2.next(), gen.nth(5));

        for _ in 0..1500 {
            let _ = gen2.next();
        }

        assert_eq!(gen2.next(), gen.nth(1500));

        let rem = gen2.len();
        for _ in 0..rem -1 {
            let _ = gen2.next();
        }

        let last = gen2.next();
        assert_eq!(last, gen.nth(rem - 1));
        assert!(last.is_some());
        assert!(gen2.next().is_none());
        assert!(gen.next().is_none());
    }

    #[test]
    fn test_tumbleweed() {
        let mut game = Tumbleweed::new(5);

        game.play(TumbleweedMove::Setup((1, 1).into(), (-1, -2).into())).unwrap();
        for (c, e) in game.los.iter_coord_fields() {
            if c == (1, 1).into() || c == (-1, -2).into(){
                continue;
            }
            match (c.x, c.y, c.z()) {
                (1, _, _) => assert_eq!(e[0], 3),
                (_, 1, _) => assert_eq!(e[0], 3),
                (_, _, -2) => assert_eq!(e[0], 3),
                _ => assert_eq!(e[0], 0),
            }

            match (c.x, c.y, c.z()) {
                (-1, _, _) => assert_eq!(e[1], 3),
                (_, -2, _) => assert_eq!(e[1], 3),
                (_, _, 3) => assert_eq!(e[1], 3),
                _ => assert_eq!(e[1], 0),
            }
        }

        assert!(game.play(TumbleweedMove::Play((2, 2).into())).is_err());
    }
}
