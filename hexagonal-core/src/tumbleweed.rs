//! Implementation of a
//! [Tumbleweed](https://boardgamegeek.com/boardgame/318702/tumbleweed)
//! game rules.
//!
//! A stack can "see" a field, when they are connected by a straight line, with
//! no stacks in between. The players take turns placing stacks of their tokens
//! on hexes that are seen by at least one friendly stack. The height of every
//! newly-placed stack equals the number of your stacks that see the new stack.
//! Replacing an existing stack with a new stack is possible, as long as the
//! new stack is taller than the previous one. This works with opponent stacks
//! (to capture), or your own stacks (to reinforce).
//!
//! Since there are no pieces at the beginning of the game, it would imply that
//! there are no valid first moves. Therefore the first move consist of the
//! first (Black) player placing one black and one white piece on the board,
//! plus the neutral stack of height 2 at the center of the board. The White
//! player can than either accept the position and continue by placing a white
//! piece wherever line-of-sight rule allows, or swap the colors, and let the
//! opponent continue from the same position.
//!
//! The game ends when no more moves can be made by either player, or after two
//! successive passes. The player who occupies over half the board wins.
//!
//! As an optimization in context of AI moves, valid moves that are no-op are
//! also disallowed. That includes placing stacks that can be immediately
//! captured, placing stack of height 6, and reinforcing stacks that are not
//! threatened.

use crate::common::{HexagonalError, HexagonalResult};
use crate::game::{Game, GameResult, Player, BW};
use crate::hexboard::{BoardCoord, Direction, DirectionIterator, RoundBoardCoord, RoundHexBoard};
use std::cell::OnceCell;
use std::cmp::Ordering;
use std::num::NonZeroU8;

#[derive(Debug, Clone)]
pub struct Tumbleweed<F = Option<ColorStack>, LS = TumbleweedLoS>
where
    F: TumbleweedField,
    LS: TumbleweedLoSField,
{
    consecutive_passes: u8,
    current: Player,
    valid_moves: OnceCell<BW<Vec<TumbleweedMove>>>,
    played_moves: Vec<TumbleweedMove>,
    board: RoundHexBoard<F>,
    los: RoundHexBoard<LS>,
}

impl<F, LS> Game for Tumbleweed<F, LS>
where
    F: TumbleweedField,
    LS: TumbleweedLoSField,
{
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
            .into()
        })[self.current]
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
                TumbleweedMove::Play(c) if self.los[c].is_valid(self.current) => true,
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

impl<F, LS> Tumbleweed<F, LS>
where
    F: TumbleweedField,
    LS: TumbleweedLoSField,
{
    #[inline]
    pub fn new(size: u8) -> Tumbleweed<F, LS> {
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
                ValidDiff::default()
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
        self.valid_moves.get_or_init(Default::default);
        {
            let oppo = self.current.opponent();
            self.valid_moves.get_mut().unwrap()[oppo].append(&mut delta[oppo]);
        }
        let valid = &mut self.valid_moves.get_mut().unwrap()[self.current];
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

        valid.append(&mut delta[self.current]);

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
        let prev_color = cur.color();
        self.board[coord] = F::new(stack, color);

        if let Some(c) = color.into() {
            update_los(&self.board, &mut self.los, c, prev_color, coord)
        } else {
            Default::default()
        }
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
        let b = self.place(TumbleweedPiece::Black, NonZeroU8::new(1).unwrap(), black);
        let w = self.place(TumbleweedPiece::White, NonZeroU8::new(1).unwrap(), white);
        [b.take_black(), w.take_white()].into()
    }

    #[inline]
    fn swap_colors(&mut self) -> ValidDiff {
        for fld in self.board.iter_fields_mut() {
            fld.swap();
        }
        for f in self.los.iter_fields_mut() {
            f.swap();
        }
        self.valid_moves.get_mut().unwrap().swap();
        Default::default()
    }

    #[inline]
    pub fn board(&self) -> &RoundHexBoard<F> {
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

/// Possible color of a Tumbleweed piece.
#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum TumbleweedPiece {
    Black,
    White,
    Neutral,
}

impl TumbleweedPiece {
    /// Similar to [Player::opponent], except [TumbleweedPiece::Neutral] returns itself
    #[inline]
    pub fn opponent(&self) -> TumbleweedPiece {
        match self {
            TumbleweedPiece::Black => TumbleweedPiece::White,
            TumbleweedPiece::White => TumbleweedPiece::Black,
            _ => *self,
        }
    }
}

impl From<Player> for TumbleweedPiece {
    /// Convert [Player] to [TumbleweedPiece]
    #[inline]
    fn from(val: Player) -> Self {
        match val {
            Player::Black => TumbleweedPiece::Black,
            Player::White => TumbleweedPiece::White,
        }
    }
}

impl From<TumbleweedPiece> for Option<Player> {
    #[inline]
    fn from(value: TumbleweedPiece) -> Self {
        match value {
            TumbleweedPiece::Neutral => None,
            TumbleweedPiece::Black => Some(Player::Black),
            TumbleweedPiece::White => Some(Player::White),
        }
    }
}

/// TumbleweedField can either be empty or has a stack of non-zero height of some color.
/// The [Default::default] constructor returns an empty field.
pub trait TumbleweedField:
    Copy + Clone + Default + std::fmt::Debug + std::hash::Hash + Send + Sync
{
    /// Creates a field populated with a stack of given height and color.
    fn new(stack: NonZeroU8, color: TumbleweedPiece) -> Self;

    /// Returns a height of a stack, or 0 if the field is empty.
    fn stack(&self) -> u8;

    /// Returns a `Some(color)` of a stack or `None` if the field is empty.
    fn color(&self) -> Option<TumbleweedPiece>;

    /// Swaps the color of the stack, leaving the height unchanged.
    fn swap(&mut self);
}

/// A struct that describes a height of a specified color and non-zero height.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorStack {
    pub stack: NonZeroU8,
    pub color: TumbleweedPiece,
}

impl TumbleweedField for Option<ColorStack> {
    #[inline]
    fn new(stack: NonZeroU8, color: TumbleweedPiece) -> Self {
        Some(ColorStack { stack, color })
    }

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

    #[inline]
    fn swap(&mut self) {
        *self = match self {
            None => None,
            Some(cs) => Some(ColorStack {
                color: cs.color.opponent(),
                stack: cs.stack,
            }),
        };
    }
}

/// Aside from the actual board state of stack of pieces on the board, there
/// are other useful properties of a game state, specifically line-of-sight
/// counts for black and white, and wether placing a stack on this field would
/// be a valid b/w move.
pub trait TumbleweedLoSField:
    Clone + std::fmt::Debug + Default + PartialEq + Eq + std::hash::Hash + Sync + Send
{
    /// Wether placing a stack on this field would be a valid move for a player
    /// of a given color.
    fn is_valid(&self, color: Player) -> bool;

    /// From how many sides is this field observed by a player of a given color.
    fn los(&self, color: Player) -> u8;

    /// Updates the valid state of this field given the current line-of-sight
    /// counts and the current occupancy state of the field. It returns an
    /// array of two booleans marking weather he field became valid in this
    /// move for Black or White player.
    #[inline]
    fn update_valid<F: TumbleweedField>(&mut self, field: &F) -> BW<bool> {
        let was_valid_b = self.is_valid(Player::Black);
        let was_valid_w = self.is_valid(Player::White);
        let lb = self.los(Player::Black);
        let lw = self.los(Player::White);

        let validb = (lb >= lw)
            && (lb < 6)
            && (lb > field.stack())
            && !(field.color() == Some(TumbleweedPiece::Black) && (lb > lw + 1));

        self.set_valid(Player::Black, validb);

        let validw = (lw >= lb)
            && (lw < 6)
            && (lw > field.stack())
            && !(field.color() == Some(TumbleweedPiece::White) && (lw > lb + 1));

        self.set_valid(Player::White, validw);

        [!was_valid_b && validb, !was_valid_w && validw].into()
    }

    /// Swaps line-of-sight and validity status for black and white players.
    fn swap(&mut self);

    #[inline]
    fn incdec(&mut self, color: Player, i: u8, d: u8) {
        self.inc(color, i);
        self.dec(color.opponent(), d);
    }

    /// Increases line-of-sight count for a given player by i.
    fn inc(&mut self, color: Player, i: u8);

    /// Decreases line-of-sight count for a given player by i.
    fn dec(&mut self, color: Player, i: u8);

    #[inline]
    fn reset_valid(&mut self) {
        self.set_valid(Player::Black, false);
        self.set_valid(Player::White, false);
    }

    /// Sets a valid status for a given player.
    fn set_valid(&mut self, color: Player, valid: bool);
}

/// A structure that implements [TumbleweedLoSField] trait while encoding
/// all the data in a single byte.
///
/// As the maximum number of sides a field can be observed is 6, which can be
/// encoded in 3 bits, and weather the field is valid or not in 1 bit, the
/// entire LoSField state for both players can be encoded into a single `u8`.
/// This however incurs computation overhead when manipulating and retrieving
/// the data. In presence of vectorized algorithms that could parallelize this
/// processing it could be worthwhile, but as it stands, the naive
/// implementation that consumes 4 bytes to store the same data is faster. The
/// good news is that the [Tumbleweed] struct is parametrized on LosField type,
/// so if an algorithm that can take an advantage of the compact representation
/// is developed, it will be trivial to switch.
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
    fn incdec(&mut self, color: Player, i: u8, d: u8) {
        let id = [(2, 32), (32, 2)][color as usize];
        self.0 = self.0 + id.0 * i - id.1 * d;
    }

    #[inline]
    fn inc(&mut self, color: Player, i: u8) {
        self.0 += [2, 32][color as usize] * i;
    }

    #[inline]
    fn dec(&mut self, color: Player, i: u8) {
        self.0 -= [2, 32][color as usize] * i;
    }

    #[inline]
    fn reset_valid(&mut self) {
        self.0 &= 0b11101110;
    }

    #[inline]
    fn set_valid(&mut self, color: Player, valid: bool) {
        self.0 &= [0b11111110, 0b11101111][color as usize];
        self.0 |= [1, 16][color as usize] * valid as u8;
    }
}

/// A simple struct that implements the [TumbleweedLoSField] trait.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TumbleweedLoS(BW<u8>, BW<bool>);

impl TumbleweedLoSField for TumbleweedLoS {
    #[inline]
    fn is_valid(&self, color: Player) -> bool {
        self.1[color]
    }

    #[inline]
    fn los(&self, color: Player) -> u8 {
        self.0[color]
    }

    #[inline]
    fn swap(&mut self) {
        self.0.swap();
        self.1.swap();
    }

    #[inline]
    fn inc(&mut self, color: Player, i: u8) {
        self.0[color] += i;
    }

    #[inline]
    fn dec(&mut self, color: Player, i: u8) {
        self.0[color] -= i;
    }

    #[inline]
    fn set_valid(&mut self, color: Player, valid: bool) {
        self.1[color] = valid;
    }
}

/// Possible kinds of moves in the game.
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub enum TumbleweedMove {
    Setup(BoardCoord, BoardCoord),
    Swap,
    Play(RoundBoardCoord),
    Pass,
}

type ValidDiff = BW<Vec<TumbleweedMove>>;

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

/// Iterator that yields all valid start moves for a given board size.
///
/// On the first move, Black player places one black and one white piece on
/// any field on the board, except the center which is reserved for the
/// neutral stack. This gives an order of n<sup>4</sup> valid moves for a
/// board of size `n`. In order to avoid materializing that list,
/// `GenStartMoves` is an iterator with an efficient [GenStartMoves::nth] implementation,
/// that can, for example, be used to select a move randomly.
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

fn update_los<F: TumbleweedField, LS: TumbleweedLoSField>(
    board: &RoundHexBoard<F>,
    los: &mut RoundHexBoard<LS>,
    color: Player,
    prev_color: Option<TumbleweedPiece>,
    coord: RoundBoardCoord,
) -> ValidDiff {
    let oppo = color.opponent();

    let coord_i = coord.idx(board);
    let coord_c = coord.coord(board);

    los[coord_i].reset_valid();

    if prev_color == Some(color.into()) {
        return Default::default();
    }

    let bs = board.size();
    let bs1 = board.size() as i8 - 1;
    let mut ret = BW::<Vec<TumbleweedMove>>::default();

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
            if let Some(color) = board[cidx].color() {
                other_piece_l = Some(color);
                break;
            }
        }

        let (mut losu, mut losd) = los_ud(other_piece_l, prev_color, oppo.into());

        let mut other_piece_r = None;
        for cc in itdown.skip(1).take((bs1 + cu).min(bs1 - cd) as usize) {
            let cidx = RoundBoardCoord::C(cc).as_idx(board);
            let field = &board[cidx];
            let losc = &mut los[cidx];
            losc.inc(color, losu);
            losc.dec(oppo, losd);

            let new_valid = losc.update_valid(field);
            for i in [Player::Black, Player::White] {
                if new_valid[i] {
                    ret[i].push(cidx.into());
                }
            }
            if let Some(color) = field.color() {
                other_piece_r = Some(color);
                break;
            }
        }

        (losu, losd) = los_ud(other_piece_r, prev_color, oppo.into());
        for c in to_update_l {
            let field = &board[c];
            let losc = &mut los[c];
            losc.inc(color, losu);
            losc.dec(oppo, losd);
            let new_valid = losc.update_valid(field);
            for i in [Player::Black, Player::White] {
                if new_valid[i] {
                    ret[i].push(c.into());
                }
            }
        }
    }

    ret
}

#[allow(dead_code)]
fn update_los_branchless<F: TumbleweedField, LS: TumbleweedLoSField>(
    board: &RoundHexBoard<F>,
    los: &mut RoundHexBoard<LS>,
    color: Player,
    prev_color: Option<TumbleweedPiece>,
    coord: RoundBoardCoord,
) -> ValidDiff {

    let oppo = color.opponent();

    let coord_i = coord.idx(board);
    let coord_c = coord.coord(board);

    los[coord_i].reset_valid();

    if prev_color == Some(color.into()) {
        return Default::default();
    }

    let bs1 = board.size() as i8 - 1;

    let (x, y, z) = (coord_c.x, coord_c.y, coord_c.z());
    let (xa, ya, za) = (x.abs(), y.abs(), z.abs());
    let l2 = board.num_rows() - 1;
    let (xl, yl, zl) = (l2 - xa as usize, l2 - ya as usize, l2 - za as usize);

    let total_affected = xl + yl + zl;
    let mut coords: Vec<RoundBoardCoord> = Vec::with_capacity(total_affected);
    let rxu = (bs1 - y).min(bs1 + z);
    let rxd = (bs1 + y).min(bs1 - z);
    let ryu = (bs1 - x).min(bs1 + z);
    let ryd = (bs1 + x).min(bs1 - z);
    let rzu = (bs1 - x).min(bs1 + y);
    let rzd = (bs1 + x).min(bs1 - y);
    let rx = rxu as usize + rxd as usize;
    let ry = ryu as usize + ryd as usize;
    let rz = rzu as usize + rzd as usize;
    let splits = [0, rxu as usize, rx, rx + ryu as usize, rx + ry, rx + ry + rzu as usize, rx + ry + rz];
    coords.extend((1..=rxu).map(|i| RoundBoardCoord::from((x, y + i))));
    coords.extend((1..=rxd).map(|i| RoundBoardCoord::from((x, y - i))));
    coords.extend((1..=ryu).map(|i| RoundBoardCoord::from((x + i, y))));
    coords.extend((1..=ryd).map(|i| RoundBoardCoord::from((x - i, y))));
    coords.extend((1..=rzu).map(|i| RoundBoardCoord::from((x + i, y - i))));
    coords.extend((1..=rzd).map(|i| RoundBoardCoord::from((x - i, y + i))));

    debug_assert_eq!(total_affected, coords.len());

    let indices: Vec<RoundBoardCoord> = coords.iter().map(|c| c.as_idx(board)).collect();
    let mut fields: Vec<F> = Vec::with_capacity(total_affected);
    let mut afidx: Vec<RoundBoardCoord> = Vec::with_capacity(total_affected);

    let mut counts = [0; 7];
    let mut uds = [(0, 0); 6];

    for i in 0..6 {
        let mut other = None;
        let mut count = splits[i + 1] - splits[i];
        for (e, &idx) in indices[splits[i]..splits[i + 1]].iter().enumerate() {
            let f = board[idx];
            fields.push(f);
            afidx.push(idx);
            if f.color().is_some() {
                count = e + 1;
                other = f.color();
                break;
            }
        }
        counts[i + 1] = counts[i] + count;
        // println!("{i} {}", i + 1 - (i % 2) * 2);
        uds[i + 1 - (i % 2) * 2] = los_ud(other, prev_color, oppo.into());
    }

    assert_eq!(counts[6], fields.len());

    let mut i = 0;
    let (mut validb, mut validw) = (0, 0);

//     let valids = for (e, (&idx, &fld)) in afidx.iter().zip(fields.iter()).enumerate() {
//         i += (counts[i + 1] == e) as usize;
//         los[idx].incdec(color, uds[i].0, uds[i].1);
//
//     }

    let valids: Vec<BW<bool>> = afidx.iter().zip(fields.iter()).enumerate().map(|(e, (&idx, &fld))|{
        i += (counts[i + 1] == e) as usize;
        i += (counts[i + 1] == e) as usize;
        los[idx].incdec(color, uds[i].0, uds[i].1);
        let new_vals = los[idx].update_valid(&fld);
        validb += new_vals[Player::Black] as usize;
        validw += new_vals[Player::White] as usize;
        new_vals
    }).collect();

    afidx.iter().zip(valids.iter()).fold([Vec::with_capacity(validb), Vec::with_capacity(validw)].into(), |mut acc, (&idx, &val)| {
        if val[Player::Black] {
            acc[Player::Black].push(idx.into())
        }
        if val[Player::White] {
            acc[Player::White].push(idx.into())
        }
        acc
    })
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
