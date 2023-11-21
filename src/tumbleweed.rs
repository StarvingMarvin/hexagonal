use crate::common::{HexagonalError, HexagonalResult};
use crate::game::{Game, GameResult, Player};
use crate::hexboard::{BoardCoord, Direction, DirectionIterator, RoundHexBoard};
use crate::{get_offsets, get_round_idx};
use std::cell::OnceCell;
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

type ValidDiff = [Vec<TumbleweedMove>; 2];

#[derive(Debug, Clone)]
pub struct Tumbleweed {
    consecutive_passes: u8,
    current: Player,
    valid_moves: OnceCell<[Vec<TumbleweedMove>; 2]>,
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
        self.valid_moves.get_or_init(|| {
            [
                GenStartMoves::new(self.board().get_coords()).collect(),
                vec![],
            ]
        })[0]
            .as_slice()
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
                _ => false,
            },
            _ => {
                !self.game_over()
                    && match mv {
                        TumbleweedMove::Pass => true,
                        TumbleweedMove::Play(c) if self.los[c][self.current as usize] % 2 > 0 => {
                            true
                        }
                        _ => false,
                    }
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
        Tumbleweed {
            consecutive_passes: 0,
            board: RoundHexBoard::new(size),
            los: RoundHexBoard::new(size),
            current: Player::Black,
            valid_moves: OnceCell::new(),
            played_moves: vec![],
        }
    }

    pub fn play_unchecked(&mut self, tmove: TumbleweedMove) {
        self.played_moves.push(tmove);

        let delta = match tmove {
            TumbleweedMove::Setup(b, w) => self.setup((0, 0).into(), b, w),
            TumbleweedMove::Swap => self.swap_colors(),
            TumbleweedMove::Play(m) => {
                let stack = self.los[m][self.current as usize] / 2;
                self.place(self.current.into(), stack, m)
            }
            TumbleweedMove::Pass => {
                self.consecutive_passes += 1;
                [vec![], vec![]]
            }
        };

        if tmove != TumbleweedMove::Pass {
            self.consecutive_passes = 0;
        }

        self.current = self.next_player();
        self.update_valids(delta);
    }

    fn update_valids(&mut self, mut delta: ValidDiff) {
        let last = &self.last_move();
        let los = &self.los;
        let coords = &self.board.get_coords();
        self.valid_moves.get_or_init(|| [vec![], vec![]]);
        {
            let oppo = self.current.opponent() as usize;
            self.valid_moves.get_mut().unwrap()[oppo].append(&mut delta[oppo]);
        }
        let valid = &mut self.valid_moves.get_mut().unwrap()[self.current as usize];
        if self.consecutive_passes >= 2 {
            valid.clear();
            return;
        }

        valid.retain(
            |m| matches!(m, TumbleweedMove::Play(c) if los[*c][self.current as usize] & 1 != 0),
        );
        valid.append(&mut delta[self.current as usize]);

        match last {
            Some(m) => {
                if let TumbleweedMove::Setup(_, _) = m {
                    valid.push(TumbleweedMove::Swap);
                } else {
                    valid.push(TumbleweedMove::Pass);
                }
            }
            None => valid.extend(GenStartMoves::new(coords)),
        };
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

    fn place(&mut self, color: TumbleweedPiece, stack: u8, coord: BoardCoord) -> ValidDiff {
        let cur = &self.board[coord];
        let prev_color = cur.color;
        self.board[coord] = TumbleweedField {
            stack,
            color: Some(color),
        };
        update_los(&self.board, &mut self.los, color, prev_color, coord)
    }

    fn setup(&mut self, neutral: BoardCoord, black: BoardCoord, white: BoardCoord) -> ValidDiff {
        self.place(TumbleweedPiece::Neutral, 2, neutral);
        let [b, _] = self.place(TumbleweedPiece::Black, 1, black);
        let [_, w] = self.place(TumbleweedPiece::White, 1, white);
        [b, w]
    }

    fn swap_colors(&mut self) -> ValidDiff {
        for f in self.board.iter_fields_mut() {
            f.color = f.color.map(|c| c.opponent());
        }
        for f in self.los.iter_fields_mut() {
            f.reverse();
        }
        self.valid_moves.get_mut().unwrap().reverse();
        [vec![], vec![]]
    }

    pub fn board(&self) -> &RoundHexBoard<TumbleweedField> {
        &self.board
    }

    pub fn los(&self) -> &RoundHexBoard<[u8; 2]> {
        &self.los
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
) -> ValidDiff {
    if color == TumbleweedPiece::Neutral {
        return [vec![], vec![]];
    }

    let oppo = color.opponent();
    let colu = color as usize;
    let opu = oppo as usize;
    los[coord][colu] &= 254;
    los[coord][opu] &= 254;

    if prev_color == Some(color) {
        return [vec![], vec![]];
    }

    let bs = board.size();
    let bs1 = board.size() as i8 - 1;
    let offsets = get_offsets(board.size() as usize);
    let mut ret = [vec![], vec![]];

    for dir in [Direction::XY, Direction::YZ, Direction::ZX].iter() {
        let mut to_update_l = Vec::with_capacity(bs as usize * 2);

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
            let cidx = get_round_idx(bs, cc, offsets);
            to_update_l.push(cidx);
            if board.as_slice()[cidx].stack > 0 {
                other_piece_l = board.as_slice()[cidx].color;
                break;
            }
        }

        let (mut losu, mut losd) = los_ud(other_piece_l, prev_color, oppo);

        let mut other_piece_r = None;
        for cc in itdown.skip(1).take((bs1 + cu).min(bs1 - cd) as usize) {
            let cidx = get_round_idx(board.size(), cc, offsets);
            let field = &board.as_slice()[cidx];
            let losc = &mut los.as_mut_slice()[cidx];
            losc[colu] += losu;
            losc[opu] -= losd;
            if valid_field(losc, field, color, oppo) {
                ret[colu].push(TumbleweedMove::Play(cc));
            }
            if field.stack > 0 {
                other_piece_r = field.color;
                break;
            }
        }

        (losu, losd) = los_ud(other_piece_r, prev_color, oppo);
        for c in to_update_l {
            let field = &board.as_slice()[c];
            let losc = &mut los.as_mut_slice()[c];
            losc[colu] += losu;
            losc[opu] -= losd;
            if valid_field(losc, field, color, oppo) {
                ret[colu].push(TumbleweedMove::Play(board.get_coords()[c]));
            }
        }
    }

    ret
}

#[inline]
fn valid_field(
    los: &mut [u8; 2],
    field: &TumbleweedField,
    color: TumbleweedPiece,
    oppo: TumbleweedPiece,
) -> bool {
    let colu = color as usize;
    let oppu = oppo as usize;
    let was_valid = los[colu] & 1;
    let lc = los[colu] / 2;
    let lo = los[oppu] / 2;
    let validc = ((lc > field.stack)
        && (lc >= lo)
        && (lc < 6)
        && !(field.color == Some(color) && (lc > lo + 1))) as u8;
    los[colu] &= 254;
    los[colu] |= validc;

    let valido = ((lo > field.stack)
        && (lo >= lc)
        && (lo < 6)
        && !(field.color == Some(oppo) && (lo > lc + 1))) as u8;

    los[oppu] &= 254;
    los[oppu] |= valido;

    validc != 0 && was_valid == 0
}

#[inline]
fn los_ud(
    other_piece: Option<TumbleweedPiece>,
    prev_color: Option<TumbleweedPiece>,
    oppo: TumbleweedPiece,
) -> (u8, u8) {
    let oppodec = (other_piece == Some(oppo) && prev_color.is_none()) || prev_color == Some(oppo);

    let cup = 2
        * (oppodec
            || other_piece.is_none()
            || other_piece == Some(TumbleweedPiece::Neutral)
            || prev_color == Some(TumbleweedPiece::Neutral)) as u8;

    let odown = 2 * oppodec as u8;
    (cup, odown)
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
        for _ in 0..rem - 1 {
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

        game.play(TumbleweedMove::Setup((1, 1).into(), (-1, -2).into()))
            .unwrap();
        for (c, e) in game.los.iter_coord_fields() {
            if c == (1, 1).into() || c == (-1, -2).into() {
                continue;
            }
            match (c.x, c.y, c.z()) {
                (1, _, _) => assert_eq!(e[0], 3, "{:?}", c),
                (_, 1, _) => assert_eq!(e[0], 3, "{:?}", c),
                (_, _, -2) => assert_eq!(e[0], 3, "{:?}", c),
                _ => assert_eq!(e[0], 0, "{:?}", c),
            }

            match (c.x, c.y, c.z()) {
                (-1, _, _) => assert_eq!(e[1], 3, "{:?}", c),
                (_, -2, _) => assert_eq!(e[1], 3, "{:?}", c),
                (_, _, 3) => assert_eq!(e[1], 3, "{:?}", c),
                _ => assert_eq!(e[1], 0, "{:?}", c),
            }
        }

        assert!(game.play(TumbleweedMove::Play((2, 2).into())).is_err());

        assert!(game.play(TumbleweedMove::Play((-1, 3).into())).is_ok());

        assert_eq!(game.los[(-1, -3).into()][1], 3);
        assert_eq!(game.los[(-1, -2).into()][1], 2);
        assert_eq!(game.los[(-1, 3).into()][1], 2);
        assert_eq!(game.los[(-1, 4).into()][1], 3);

        for i in -1..3 {
            assert_eq!(game.los[(-1, i).into()][1], 5);
        }
    }
}
