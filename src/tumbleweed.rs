use crate::common::{HexagonalError, HexagonalResult};
use crate::game::{Game, GameResult, Player};
use crate::hexboard::{BoardCoord, Direction, DirectionIterator, RoundBoardCoord, RoundHexBoard};
use std::cell::OnceCell;
use std::cmp::Ordering;
use std::num::NonZeroU8;

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

trait MaybeField {
    fn stack(&self) -> u8;
    fn color(&self) -> Option<TumbleweedPiece>;
}

impl MaybeField for Option<TumbleweedField> {
    #[inline]
    fn stack(&self) -> u8 {
        match self {
            None => 0,
            Some(x) => x.stack.into(),
        }
    }

    #[inline]
    fn color(&self) -> Option<TumbleweedPiece> {
        self.as_ref().map(|f| f.color)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TumbleweedField {
    pub stack: NonZeroU8,
    pub color: TumbleweedPiece,
}

pub trait TumbleweedLoSField:
    Clone + std::fmt::Debug + Default + PartialEq + Eq + std::hash::Hash + Sync + Send
{
    fn is_valid(&self, color: Player) -> bool;

    fn los(&self, color: Player) -> u8;

    #[inline]
    fn update_valid(&mut self, field: &Option<TumbleweedField>) -> [bool; 2] {
        let was_valid_b = self.is_valid(Player::Black);
        let was_valid_w = self.is_valid(Player::White);
        let lb = self.los(Player::Black);
        let lw = self.los(Player::White);

        let validb = (lb >= lw)
            && (lb < 6)
            && match field {
                None => true,
                Some(f) => {
                    (lb > f.stack.get()) && !(f.color == TumbleweedPiece::Black && (lb > lw + 1))
                }
            };

        self.set_valid(Player::Black, validb);

        let validw = (lw >= lb)
            && (lw < 6)
            && match field {
                None => true,
                Some(f) => {
                    (lw > f.stack.get()) && !(f.color == TumbleweedPiece::White && (lw > lb + 1))
                }
            };

        self.set_valid(Player::White, validw);

        [!was_valid_b && validb, !was_valid_w && validw]
    }

    fn swap(&mut self);

    fn inc(&mut self, color: TumbleweedPiece, i: u8);

    fn dec(&mut self, color: TumbleweedPiece, i: u8);

    fn reset_valid(&mut self, color: Player);

    fn set_valid(&mut self, color: Player, valid: bool);
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TumbleweedCompactLoS(u8);

impl TumbleweedLoSField for TumbleweedCompactLoS {
    #[inline]
    fn is_valid(&self, color: Player) -> bool {
        self.0 & [1, 16][color as usize] > 0
    }

    #[inline]
    fn los(&self, color: Player) -> u8 {
        self.0 >> [1, 5][color as usize] & 0b0111
    }

    #[inline]
    fn swap(&mut self) {
        self.0 = (self.0 & 0b11110000 >> 4) | (self.0 << 4);
    }

    #[inline]
    fn inc(&mut self, color: TumbleweedPiece, i: u8) {
        self.0 += [2, 32][color as usize] * i;
    }

    #[inline]
    fn dec(&mut self, color: TumbleweedPiece, i: u8) {
        self.0 -= [2, 32][color as usize] * i;
    }

    #[inline]
    fn reset_valid(&mut self, color: Player) {
        self.0 &= [0b11111110, 0b11101111][color as usize];
    }

    #[inline]
    fn set_valid(&mut self, color: Player, valid: bool) {
        self.0 &= [0b11111110, 0b11101111][color as usize];
        self.0 |= [1, 16][color as usize] * valid as u8;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TumbleweedLoS([u8; 2], [bool; 2]);

impl TumbleweedLoSField for TumbleweedLoS {
    #[inline]
    fn is_valid(&self, color: Player) -> bool {
        self.1[color as usize]
    }

    #[inline]
    fn los(&self, color: Player) -> u8 {
        self.0[color as usize]
    }

    #[inline]
    fn swap(&mut self) {
        self.0.reverse();
        self.1.reverse();
    }

    #[inline]
    fn inc(&mut self, color: TumbleweedPiece, i: u8) {
        self.0[color as usize] += i;
    }

    #[inline]
    fn dec(&mut self, color: TumbleweedPiece, i: u8) {
        self.0[color as usize] -= i;
    }

    #[inline]
    fn reset_valid(&mut self, color: Player) {
        self.1[color as usize] = false;
    }

    #[inline]
    fn set_valid(&mut self, color: Player, valid: bool) {
        self.1[color as usize] = valid;
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub enum TumbleweedMove {
    Setup(BoardCoord, BoardCoord),
    Swap,
    Play(RoundBoardCoord),
    Pass,
}

impl From<BoardCoord> for TumbleweedMove {
    #[inline]
    fn from(value: BoardCoord) -> Self {
        TumbleweedMove::Play(value.into())
    }
}

impl From<usize> for TumbleweedMove {
    #[inline]
    fn from(value: usize) -> Self {
        TumbleweedMove::Play(value.into())
    }
}

impl From<RoundBoardCoord> for TumbleweedMove {
    #[inline]
    fn from(value: RoundBoardCoord) -> Self {
        TumbleweedMove::Play(value)
    }
}

impl From<(i8, i8)> for TumbleweedMove {
    #[inline]
    fn from(value: (i8, i8)) -> Self {
        TumbleweedMove::Play(BoardCoord::from(value).into())
    }
}

impl From<(BoardCoord, BoardCoord)> for TumbleweedMove {
    #[inline]
    fn from((b, w): (BoardCoord, BoardCoord)) -> Self {
        TumbleweedMove::Setup(b, w)
    }
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
            let ret = (self.coords[self.b], self.coords[self.w]).into();
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

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.coords.len();
        let total = (l - 1) * (l - 2);
        (total - self.cnt, Some(total - self.cnt))
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.cnt += n;
        let l2 = self.coords.len() - 2;
        self.b = self.cnt / l2;
        self.b += (self.b >= self.zero) as usize;

        self.w = self.cnt % l2;
        self.w += (self.w >= self.b) as usize + (self.w >= self.zero) as usize;

        self.next()
    }
}

impl ExactSizeIterator for GenStartMoves {}

type ValidDiff = [Vec<TumbleweedMove>; 2];

#[derive(Debug, Clone)]
pub struct Tumbleweed<LS = TumbleweedLoS>
where
    LS: TumbleweedLoSField,
{
    consecutive_passes: u8,
    current: Player,
    valid_moves: OnceCell<[Vec<TumbleweedMove>; 2]>,
    played_moves: Vec<TumbleweedMove>,
    board: RoundHexBoard<Option<TumbleweedField>>,
    los: RoundHexBoard<LS>,
}

impl<LS: TumbleweedLoSField> Game for Tumbleweed<LS> {
    type Move = TumbleweedMove;

    #[inline]
    fn game_over(&self) -> bool {
        self.consecutive_passes >= 2
    }

    #[inline]
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
        })[self.current as usize]
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
                TumbleweedMove::Play(c) if self.los[c].is_valid(Player::White) => true,
                _ => false,
            },
            _ => {
                !self.game_over()
                    && match mv {
                        TumbleweedMove::Pass => true,
                        TumbleweedMove::Play(c) if self.los[c].is_valid(self.current) => true,
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

    #[inline]
    fn current_player(&self) -> Player {
        self.current
    }

    #[inline]
    fn next_player(&self) -> Player {
        self.current.opponent()
    }

    #[inline]
    fn last_move(&self) -> Option<TumbleweedMove> {
        self.played_moves.last().copied()
    }
}

impl<LS: TumbleweedLoSField> Tumbleweed<LS> {
    #[inline]
    pub fn new(size: u8) -> Tumbleweed<LS> {
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
            TumbleweedMove::Setup(b, w) => {
                self.setup(BoardCoord::new(0, 0).into(), b.into(), w.into())
            }
            TumbleweedMove::Swap => self.swap_colors(),
            TumbleweedMove::Play(m) => {
                let stack = self.los[m].los(self.current);
                self.place(self.current.into(), NonZeroU8::new(stack).unwrap(), m)
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

    #[inline]
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

        for i in (0..valid.len()).rev() {
            let m = valid[i];
            if !matches!(m, TumbleweedMove::Play(c) if los[c].is_valid(self.current)) {
                valid.swap_remove(i);
            }
        }

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
                let lb = los.los(Player::Black);
                let lw = los.los(Player::White);
                ((lb > lw) && (lb > field.stack()))
                    || ((field.color() == Some(TumbleweedPiece::Black)) && (field.stack() >= lw))
            })
            .count() as i16;

        let pointsw = self
            .los
            .iter_fields()
            .zip(self.board.iter_fields())
            .filter(|(los, field)| {
                let lb = los.los(Player::Black);
                let lw = los.los(Player::White);
                ((lw > lb) && (lw > field.stack()))
                    || ((field.color() == Some(TumbleweedPiece::White)) && (field.stack() >= lb))
            })
            .count() as i16;

        (pointsb, pointsw)
    }

    fn place(
        &mut self,
        color: TumbleweedPiece,
        stack: NonZeroU8,
        coord: RoundBoardCoord,
    ) -> ValidDiff {
        let coord = coord.as_idx(&self.board);
        let cur = &self.board[coord];
        let prev_color = cur.as_ref().map(|f| f.color);
        self.board[coord] = Some(TumbleweedField { stack, color });
        update_los(&self.board, &mut self.los, color, prev_color, coord)
    }

    #[inline]
    fn setup(
        &mut self,
        neutral: RoundBoardCoord,
        black: RoundBoardCoord,
        white: RoundBoardCoord,
    ) -> ValidDiff {
        self.place(
            TumbleweedPiece::Neutral,
            NonZeroU8::new(2).unwrap(),
            neutral,
        );
        let [b, _] = self.place(TumbleweedPiece::Black, NonZeroU8::new(1).unwrap(), black);
        let [_, w] = self.place(TumbleweedPiece::White, NonZeroU8::new(1).unwrap(), white);
        [b, w]
    }

    #[inline]
    fn swap_colors(&mut self) -> ValidDiff {
        for fld in self.board.iter_fields_mut() {
            fld.as_mut().map(|f| f.color = f.color.opponent());
        }
        for f in self.los.iter_fields_mut() {
            f.swap();
        }
        self.valid_moves.get_mut().unwrap().reverse();
        [vec![], vec![]]
    }

    #[inline]
    pub fn board(&self) -> &RoundHexBoard<Option<TumbleweedField>> {
        &self.board
    }

    #[inline]
    pub fn los(&self) -> &RoundHexBoard<LS> {
        &self.los
    }

    #[inline]
    pub fn played_moves(&self) -> &[TumbleweedMove] {
        &self.played_moves
    }
}

fn update_los<LS: TumbleweedLoSField>(
    board: &RoundHexBoard<Option<TumbleweedField>>,
    los: &mut RoundHexBoard<LS>,
    color: TumbleweedPiece,
    prev_color: Option<TumbleweedPiece>,
    coord: RoundBoardCoord,
) -> ValidDiff {
    if color == TumbleweedPiece::Neutral {
        return [vec![], vec![]];
    }

    let oppo = color.opponent();

    let coord_i = coord.idx(board);
    let coord_c = coord.coord(board);

    los[coord_i].reset_valid(Player::Black);
    los[coord_i].reset_valid(Player::White);

    if prev_color == Some(color) {
        return [vec![], vec![]];
    }

    let bs = board.size();
    let bs1 = board.size() as i8 - 1;
    let mut ret = [vec![], vec![]];

    for dir in [Direction::XY, Direction::YZ, Direction::ZX].iter() {
        let mut to_update_l = Vec::with_capacity(bs as usize * 2);

        let itup = DirectionIterator::new(coord_c, *dir);
        let itdown = DirectionIterator::new(coord_c, -*dir);

        let (cu, cd) = match dir {
            Direction::YZ => (coord_c.y, coord_c.z()),
            Direction::XY => (coord_c.x, coord_c.y),
            Direction::ZX => (coord_c.z(), coord_c.x),
            _ => panic!(),
        };

        let mut other_piece_l = None;
        for cc in itup.skip(1).take((bs1 - cu).min(bs1 + cd) as usize) {
            let cidx = RoundBoardCoord::C(cc).as_idx(board);
            to_update_l.push(cidx);
            if let Some(field) = board[cidx] {
                other_piece_l = Some(field.color);
                break;
            }
        }

        let (mut losu, mut losd) = los_ud(other_piece_l, prev_color, oppo);

        let mut other_piece_r = None;
        for cc in itdown.skip(1).take((bs1 + cu).min(bs1 - cd) as usize) {
            let cidx = RoundBoardCoord::C(cc).as_idx(board);
            let field = &board[cidx];
            let losc = &mut los[cidx];
            losc.inc(color, losu);
            losc.dec(oppo, losd);

            let new_valid = losc.update_valid(field);
            for i in 0..2 {
                if new_valid[i] {
                    ret[i].push(cidx.into());
                }
            }
            if let Some(f) = field {
                other_piece_r = Some(f.color);
                break;
            }
        }

        (losu, losd) = los_ud(other_piece_r, prev_color, oppo);
        for c in to_update_l {
            let field = &board[c];
            let losc = &mut los[c];
            losc.inc(color, losu);
            losc.dec(oppo, losd);
            let new_valid = losc.update_valid(field);
            for i in 0..2 {
                if new_valid[i] {
                    ret[i].push(c.into());
                }
            }
        }
    }

    ret
}

#[inline]
fn los_ud(
    other_piece: Option<TumbleweedPiece>,
    prev_color: Option<TumbleweedPiece>,
    oppo: TumbleweedPiece,
) -> (u8, u8) {
    let oppodec = (other_piece == Some(oppo) && prev_color.is_none()) || prev_color == Some(oppo);

    let cup = (oppodec
        || other_piece.is_none()
        || other_piece == Some(TumbleweedPiece::Neutral)
        || prev_color == Some(TumbleweedPiece::Neutral)) as u8;

    let odown = oppodec as u8;
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
    fn test_los() {
        let mut game: Tumbleweed = Tumbleweed::new(5);

        game.play(TumbleweedMove::Setup((1, 1).into(), (-1, -2).into()))
            .unwrap();
        for (c, e) in game.los.iter_coord_fields() {
            if c == (1, 1).into() || c == (-1, -2).into() {
                continue;
            }
            match (c.x, c.y, c.z()) {
                (1, _, _) | (_, 1, _) | (_, _, -2) => {
                    assert!(e.is_valid(Player::Black));
                    assert_eq!(e.los(Player::Black), 1);
                }
                _ => {
                    assert!(!e.is_valid(Player::Black));
                    assert_eq!(e.los(Player::Black), 0);
                }
            }

            match (c.x, c.y, c.z()) {
                (-1, _, _) | (_, -2, _) | (_, _, 3) => {
                    assert!(e.is_valid(Player::White));
                    assert_eq!(e.los(Player::White), 1);
                }
                _ => {
                    assert!(!e.is_valid(Player::White));
                    assert_eq!(e.los(Player::White), 0);
                }
            }
        }

        assert!(game.play((2, 2).into()).is_err());

        assert!(game.play((-1, 3).into()).is_ok());

        assert!(game.los[(-1, -3)].is_valid(Player::White));
        assert_eq!(game.los[(-1, -3)].los(Player::White), 1);

        assert!(!game.los[(-1, -2)].is_valid(Player::White));
        assert_eq!(game.los[(-1, -2)].los(Player::White), 1);

        assert!(!game.los[(-1, 3)].is_valid(Player::White));
        assert_eq!(game.los[(-1, 3)].los(Player::White), 1);

        assert!(game.los[(-1, 4)].is_valid(Player::White));
        assert_eq!(game.los[(-1, 4)].los(Player::White), 1);

        for i in -1..3 {
            assert!(game.los[(-1, i)].is_valid(Player::White));
            assert_eq!(game.los[(-1, i)].los(Player::White), 2);
        }
    }

    #[test]
    fn test_tumbleweed() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut game: Tumbleweed = Tumbleweed::new(12);
        let setup = GenStartMoves::new(game.board().get_coords())
            .choose(&mut rng)
            .unwrap();
        assert!(game.play(setup).is_ok());

        while !game.game_over() {
            let valid = game.valid_moves().to_owned();

            //check that all moves in 'valid_moves' are valid
            for mv in &valid {
                assert!(game.is_valid(*mv));
            }

            // check that none of the moves outslide of 'valid_moves' are valid
            for c in 0..game.board.num_fields() {
                let mv: TumbleweedMove = c.into();
                if !valid.contains(&mv) {
                    assert!(!game.is_valid(mv));
                    assert!(game.play(mv).is_err());
                }
            }
            let mv: TumbleweedMove = *valid.choose(&mut rng).unwrap();
            match mv {
                TumbleweedMove::Swap | TumbleweedMove::Pass if valid.len() > 1 => {
                    continue;
                }
                _ => (),
            }
            assert!(game.play(mv).is_ok());
        }
    }
}
