use crate::common::{HexagonalError, HexagonalResult};
use hex2d::Coordinate;
pub use hex2d::Direction;
use std::hash::Hash;
use std::ops::{Deref, DerefMut, Index, IndexMut};

pub trait FieldTrait: PartialEq + Clone + Hash + std::fmt::Debug + Default + Send + Sync {}
impl<T> FieldTrait for T where T: PartialEq + Clone + Hash + std::fmt::Debug + Default + Send + Sync {}

pub type BoardCoord = Coordinate<i8>;

pub trait RoundBoardIndex:
    Clone + std::fmt::Debug + Send + Sync + Hash + Copy + PartialEq + Eq
{
    fn index<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> &T;
    fn index_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> &mut T;
    fn get<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> Option<&T>;
    fn get_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> Option<&mut T>;
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum RoundBoardCoord {
    C(BoardCoord),
    I(usize),
}

impl RoundBoardCoord {
    #[inline]
    pub fn as_coord<T: FieldTrait>(&self, board: &RoundHexBoard<T>) -> Self {
        match self {
            Self::C(_c) => *self,
            Self::I(i) => Self::C(board.get_coords()[*i]),
        }
    }

    #[inline]
    pub fn as_idx<T: FieldTrait>(&self, board: &RoundHexBoard<T>) -> Self {
        match self {
            Self::C(c) => {
                let idx = get_round_idx(board.size, *c, board.offsets);
                Self::I(idx)
            }
            Self::I(_i) => *self,
        }
    }

    #[inline]
    pub fn coord<T: FieldTrait>(&self, board: &RoundHexBoard<T>) -> BoardCoord {
        match self {
            Self::C(c) => *c,
            Self::I(i) => board.get_coords()[*i],
        }
    }

    #[inline]
    pub fn idx<T: FieldTrait>(&self, board: &RoundHexBoard<T>) -> usize {
        match self {
            Self::C(c) => get_round_idx(board.size, *c, board.offsets),
            Self::I(i) => *i,
        }
    }
}

// RoundHexBoard saves all the data into a contiguous slice so it needs to
// resolve an (x, y) coordinate into the slice index. Since not every row has
// the same nuber of columns, this array saves starting offsets for each row
// for board sizes 1-127. Offsets can be retrieved from a flat array because
// of the following observations:
//
//     * number of rows for the board of (side) size is 2s - 1
//     * sum{1..n}(2s - 1) = n^2
//     * therefore offsets for row starts for board of size s is located in
//       OFFSETS[(s - 1)^2..s^2]
//
// Finally a `get_round_idx` translates (x, y) coordinate to index of a flat
// array by getting xth offset for board size s and adding y to that.
static OFFSETS: [u16; 128 * 128] = init_offsets();

const fn init_offsets() -> [u16; 128 * 128] {
    let mut res = [0; 128 * 128];
    let mut i: i16 = 1;
    while i < 128 {
        let mut cnt = 0;
        let i_1 = i - 1;
        let start = i_1 * i_1;
        let mut a = -i_1;
        while a <= i {
            res[(start + a + i_1) as usize] = cnt;
            cnt += (2 * i - a.abs() - 1) as u16;
            a += 1;
        }
        i += 1;
    }
    res
}

#[inline]
pub fn get_offsets(s: usize) -> &'static [u16] {
    &OFFSETS[(s - 1) * (s - 1)..s * s]
}

#[inline]
pub fn get_round_idx(board_size: u8, index: BoardCoord, offsets: &[u16]) -> usize {
    let is1 = board_size as i16 - 1;
    let ux = (index.x as i16 + is1) as usize;
    // ix = index.x if x < 0 else 0
    let ix = ((index.x >> 7) & index.x) as i16;
    let uy = (ix + index.y as i16 + is1) as usize;
    offsets[ux] as usize + uy
}

const SIZES: [usize; 127] = init_sizes();

const fn init_sizes() -> [usize; 127] {
    let mut ret = [0; 127];
    let mut i = 1;
    while i < 127 {
        ret[i] = i * (i - 1) * 3 + 1;
        i += 1;
    }
    ret
}

pub type IterField<'a, T> = std::slice::Iter<'a, T>;
pub type IterFieldMut<'a, T> = std::slice::IterMut<'a, T>;
pub type IntoIter<T> = std::vec::IntoIter<T>;

pub type IterCoord<'a> = std::slice::Iter<'a, BoardCoord>;

// Is there a better way to do this?
static mut COORDS: [Option<Box<[BoardCoord]>>; 127] = init_coords();
const fn init_coords() -> [Option<Box<[BoardCoord]>>; 127] {
    unsafe {
        const SIZE: usize = std::mem::size_of::<[Option<Box<[BoardCoord]>>; 127]>();
        let ret = [0_u8; SIZE];
        std::mem::transmute::<[u8; SIZE], [Option<Box<[BoardCoord]>>; 127]>(ret)
    }
}

pub fn get_coords(board_size: u8) -> &'static [BoardCoord] {
    let s = board_size as usize;
    unsafe {
        let c = &mut COORDS[s];
        c.get_or_insert_with(|| {
            let mut v = Vec::with_capacity(3 * s * (s - 1) + 1);
            let is1 = board_size as i8 - 1;

            for i in -is1..=is1 {
                let (lb, ub) = if i < 0 {
                    (-is1 - i, is1)
                } else {
                    (-is1, is1 - i)
                };

                for j in lb..=ub {
                    v.push((i, j).into());
                }
            }
            v.into_boxed_slice()
        })
    }
}

// This is roughly equivalent to coords.iter().zip(fielrs.itre()),
// but the zip is actually slightly faster so it probably shouldn't
// be used in a hot path.
pub struct IterCoordField<'a, T> {
    fields: &'a [T],
    coords: &'a [BoardCoord],
    idx: usize,
}

impl<'a, T> Iterator for IterCoordField<'a, T> {
    type Item = (BoardCoord, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.fields.len() {
            let ret = (self.coords[self.idx], &self.fields[self.idx]);
            self.idx += 1;
            Some(ret)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let ret = self.fields.len() - self.idx;
        (ret, Some(ret))
    }
}

impl<'a, T> ExactSizeIterator for IterCoordField<'a, T> {}

pub struct IterNeighbours<'a, T>
where
    T: FieldTrait,
{
    board: &'a RoundHexBoard<T>,
    ncoord: [BoardCoord; 6],
    idx: usize,
}

impl<'a, T> Iterator for IterNeighbours<'a, T>
where
    T: FieldTrait,
{
    type Item = (BoardCoord, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < 6 && !self.board.valid_coord(self.ncoord[self.idx]) {
            self.idx += 1;
        }

        if self.idx < 6 {
            let c = self.ncoord[self.idx];
            self.idx += 1;
            Some((c, &self.board[c]))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(6 - self.idx))
    }
}

#[inline]
fn max_coord(index: BoardCoord) -> u8 {
    index.x.abs().max(index.y.abs()).max(index.z().abs()) as u8
}

/// A 'round' (actually hexagonal) game board used in games like Havannah and Tumbleweed.
/// Board size is a length of the side, so a board of size 5 will have `2s - 1 = 11` rows
/// and `3s^2 - 3s + 1 = 61` fields. A board will initialize a contiguous slice of an
/// appropriate size on the heap and fill it with field type's default value. It is
/// impossible to grow or shrink the size of an already initialized board.
///
/// A board is indexed using [cube coordinates](https://www.redblobgames.com/grids/hexagons/#coordinates-cube)
/// with a (0, 0) coordinate placed on a central field. Therefore a valid set of coordinates are
/// those which respect invariance that max offset from center for any of the three coordinates
/// (third being `z = -x - y`) is less than board size: `max(|x|, |y|, |-x - y|) < size`.
/// There are two sets of methods for acceccing fields: [`RoundHexBoard::get`] and
/// [`RoundHexBoard::get_mut`] return an [`Option<&T>`] or [`Option<&mut T>`] respectively, whose
/// value is `None` if coordinate is not valid for a given board size.
/// `RoundHexBoard` also implement [`Index`] and [`IndexMut`] traits.
/// Acceccing a field using `[]` or `[]=` syntax is undefined for invalid coordinates
/// (it might panic or it might return the wrong field).
///
///```
///use hexagonal::hexboard::*;
///
///let mut board = RoundHexBoard::<i32>::new(5);
///assert_eq!(board[(1, 2)], 0);
///assert_eq!(board.get((3, 3)), None);
///board[(-2, 3)] = 4;
///assert_eq!(board.get((-2, 3)), Some(&4));
///```
#[derive(Debug, Clone)]
pub struct RoundHexBoard<T>
where
    T: FieldTrait,
{
    size: u8,
    board: Box<[T]>,
    offsets: &'static [u16],
}

impl<T> RoundHexBoard<T>
where
    T: FieldTrait,
{
    /// Constructs a new instance of a given size. In all anticipated scenarios
    /// size will be in roughly 6-12 range so instead of returning Result,
    /// the method will panic if size is greater than 127, which is well above
    /// any number that should be encountered.
    pub fn new(size: u8) -> RoundHexBoard<T> {
        if size > 127 {
            panic!("size must be less than 128");
        }
        let s = size as usize;
        let len = SIZES[s];
        let board = vec![T::default(); len];

        RoundHexBoard {
            size,
            board: board.into_boxed_slice(),
            offsets: get_offsets(s),
        }
    }

    /// Chechks if coordinate is falid for this board. Since this board type
    /// uses [cube coordinates](https://www.redblobgames.com/grids/hexagons/#coordinates-cube)
    /// with the origin at central field, coordinate is valid if it's less than
    /// `size` away along any axis: `max(|x|, |y|, |-x - y|) < size`.
    #[inline]
    pub fn valid_coord(&self, coord: BoardCoord) -> bool {
        max_coord(coord) < self.size
    }

    /// Board size, that is length in fields of a side of the hexagon that is this board.
    #[inline]
    pub fn size(&self) -> u8 {
        self.size
    }

    /// Number of rows. The shortest distance in fields between two parallel sides of the board.
    #[inline]
    pub fn num_rows(&self) -> usize {
        2 * self.size as usize - 1
    }

    /// Total number of fields on the board.
    #[inline]
    pub fn num_fields(&self) -> usize {
        self.board.len()
    }

    /// An iterator over the board fields.
    #[inline]
    pub fn iter_fields(&self) -> IterField<'_, T> {
        self.board.iter()
    }

    /// A mutable iterator over the board fields.
    #[inline]
    pub fn iter_fields_mut(&mut self) -> IterFieldMut<'_, T> {
        self.board.iter_mut()
    }

    /// An iterator over the valid board coordinates.
    #[inline]
    pub fn iter_coords(&self) -> IterCoord {
        get_coords(self.size).iter()
    }

    #[inline]
    pub fn get_coords(&self) -> &'static [BoardCoord] {
        get_coords(self.size())
    }

    #[inline]
    pub fn get_offsets(&self) -> &'static [u16] {
        self.offsets
    }

    /// An iterator over the `(BoardCoord, &T)` touples.
    #[inline]
    pub fn iter_coord_fields(&self) -> IterCoordField<'_, T> {
        IterCoordField {
            fields: &self.board,
            coords: get_coords(self.size),
            idx: 0,
        }
    }

    /// Iterate over valid neighbouring fields. It starts from [YZ][Direction::YZ] direction and goes clockwise.
    ///
    ///```
    ///let b: hexagonal::RoundHexBoard<i32> = (0..91).collect::<Box<[i32]>>().try_into().unwrap();
    ///let mut nit = b.iter_neighbours((2, 3).into());
    ///let mut nxt = nit.next().unwrap();
    ///assert_eq!((nxt.0.x, nxt.0.y, *nxt.1), (3, 2, 77));
    ///nxt = nit.next().unwrap();
    ///assert_eq!((nxt.0.x, nxt.0.y, *nxt.1), (2, 2, 68));
    ///nxt = nit.next().unwrap();
    ///assert_eq!((nxt.0.x, nxt.0.y, *nxt.1), (1, 3, 59));
    ///nxt = nit.next().unwrap();
    ///assert_eq!((nxt.0.x, nxt.0.y, *nxt.1), (1, 4, 60));
    ///assert_eq!(nit.next(), None);
    ///```
    #[inline]
    pub fn iter_neighbours(&self, coord: BoardCoord) -> IterNeighbours<T> {
        IterNeighbours {
            board: self,
            ncoord: coord.neighbors(),
            idx: 0,
        }
    }

    /// A safe way to access a field. If coordinate is not valid returns None.
    #[inline]
    pub fn get<I: RoundBoardIndex>(&self, index: I) -> Option<&T> {
        index.get(self)
    }

    /// A safe way to access a field. If coordinate is not valid returns None.
    #[inline]
    pub fn get_mut<I: RoundBoardIndex>(&mut self, index: I) -> Option<&mut T> {
        index.get_mut(self)
    }

    /// A slice view of a board data.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.board
    }

    /// A mutable slice view of a board data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.board
    }
}

impl<T: FieldTrait> TryFrom<Box<[T]>> for RoundHexBoard<T> {
    type Error = HexagonalError;

    #[inline]
    fn try_from(value: Box<[T]>) -> HexagonalResult<Self> {
        let size = SIZES.binary_search(&value.len());
        match size {
            Ok(s) => Ok(RoundHexBoard {
                size: s as u8,
                board: value,
                offsets: get_offsets(s),
            }),
            Err(_) => Err(HexagonalError::new(format!(
                "Invalid slice length {} for RoundHexBoard",
                value.len()
            ))),
        }
    }
}

impl<T: FieldTrait> TryFrom<&[T]> for RoundHexBoard<T> {
    type Error = HexagonalError;

    #[inline]
    fn try_from(value: &[T]) -> HexagonalResult<Self> {
        let size = SIZES.binary_search(&value.len());
        match size {
            Ok(s) => Ok(RoundHexBoard {
                size: s as u8,
                board: value.into(),
                offsets: get_offsets(s),
            }),
            Err(_) => Err(HexagonalError::new(format!(
                "Invalid slice length {} for RoundHexBoard",
                value.len()
            ))),
        }
    }
}

impl<T: FieldTrait> Deref for RoundHexBoard<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.board
    }
}

impl<T: FieldTrait> DerefMut for RoundHexBoard<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.board
    }
}

impl<T, I> Index<I> for RoundHexBoard<T>
where
    T: FieldTrait,
    I: RoundBoardIndex,
{
    type Output = T;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        index.index(self)
    }
}

impl<T, I> IndexMut<I> for RoundHexBoard<T>
where
    T: FieldTrait,
    I: RoundBoardIndex,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        index.index_mut(self)
    }
}

impl<T: FieldTrait> IntoIterator for RoundHexBoard<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Into::<Vec<T>>::into(self.board).into_iter()
    }
}

impl RoundBoardIndex for RoundBoardCoord {
    #[inline]
    fn index<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> &T {
        match self {
            Self::C(c) => &board[c],
            Self::I(i) => &board[i],
        }
    }

    #[inline]
    fn index_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> &mut T {
        match self {
            Self::C(c) => board.index_mut(c),
            Self::I(i) => board.index_mut(i),
        }
    }

    #[inline]
    fn get<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> Option<&T> {
        match self {
            Self::C(c) => board.get(c),
            Self::I(i) => board.get(i),
        }
    }

    #[inline]
    fn get_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> Option<&mut T> {
        match self {
            Self::C(c) => board.get_mut(c),
            Self::I(i) => board.get_mut(i),
        }
    }
}

impl From<BoardCoord> for RoundBoardCoord {
    #[inline]
    fn from(value: BoardCoord) -> Self {
        Self::C(value)
    }
}

impl From<usize> for RoundBoardCoord {
    #[inline]
    fn from(value: usize) -> Self {
        Self::I(value)
    }
}

impl RoundBoardIndex for BoardCoord {
    #[inline]
    fn index<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> &T {
        let idx = get_round_idx(board.size, self, board.offsets);
        &board.as_slice()[idx]
    }

    #[inline]
    fn index_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> &mut T {
        let idx = get_round_idx(board.size, self, board.offsets);
        &mut board.as_mut_slice()[idx]
    }

    #[inline]
    fn get<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> Option<&T> {
        board.valid_coord(self).then(|| self.index(board))
    }

    #[inline]
    fn get_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> Option<&mut T> {
        board.valid_coord(self).then(|| self.index_mut(board))
    }
}

impl RoundBoardIndex for usize {
    #[inline]
    fn index<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> &T {
        &board.as_slice()[self]
    }

    #[inline]
    fn index_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> &mut T {
        &mut board.as_mut_slice()[self]
    }

    #[inline]
    fn get<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> Option<&T> {
        (self < board.num_fields()).then(|| self.index(board))
    }

    #[inline]
    fn get_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> Option<&mut T> {
        (self < board.num_fields()).then(|| self.index_mut(board))
    }
}

impl RoundBoardIndex for (i8, i8) {
    #[inline]
    fn index<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> &T {
        let idx = get_round_idx(board.size, self.into(), board.offsets);
        &board.as_slice()[idx]
    }

    #[inline]
    fn index_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> &mut T {
        let idx = get_round_idx(board.size, self.into(), board.offsets);
        &mut board.as_mut_slice()[idx]
    }

    #[inline]
    fn get<T: FieldTrait>(self, board: &RoundHexBoard<T>) -> Option<&T> {
        board.valid_coord(self.into()).then(|| self.index(board))
    }

    #[inline]
    fn get_mut<T: FieldTrait>(self, board: &mut RoundHexBoard<T>) -> Option<&mut T> {
        board
            .valid_coord(self.into())
            .then(|| self.index_mut(board))
    }
}

macro_rules! impl_from_arr {
    ($size:expr) => {
        impl<T: FieldTrait> From<[T; SIZES[$size]]> for RoundHexBoard<T> {
            #[inline]
            fn from(value: [T; SIZES[$size]]) -> Self {
                RoundHexBoard {
                    size: $size,
                    board: Box::new(value),
                    offsets: get_offsets($size),
                }
            }
        }
    };
}

impl_from_arr! {2}
impl_from_arr! {3}
impl_from_arr! {4}
impl_from_arr! {5}
impl_from_arr! {6}
impl_from_arr! {7}
impl_from_arr! {8}
impl_from_arr! {9}
impl_from_arr! {10}
impl_from_arr! {11}
impl_from_arr! {12}
impl_from_arr! {13}
impl_from_arr! {14}
impl_from_arr! {15}
impl_from_arr! {16}

/// Generate successive coordinates in a given direction, from a given coordinate (inclusive).
/// This is an ifinite iterator.
///
///```
///use hex2d::Direction;
///use hexagonal::DirectionIterator;
///let mut di = DirectionIterator::new((2, 3).into(), Direction::XZ);
///let mut nxt = di.next().unwrap();
///assert_eq!((nxt.x, nxt.y, nxt.z()), (2, 3, -5));
///nxt = di.next().unwrap();
///assert_eq!((nxt.x, nxt.y, nxt.z()), (3, 3, -6));
///nxt = di.next().unwrap();
///assert_eq!((nxt.x, nxt.y, nxt.z()), (4, 3, -7));
///```
pub struct DirectionIterator {
    coord: BoardCoord,
    dir: Direction,
}

impl Iterator for DirectionIterator {
    type Item = BoardCoord;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.coord;
        self.coord = self.coord + self.dir;
        Some(ret)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

impl DirectionIterator {
    #[inline]
    pub fn new(start: BoardCoord, dir: Direction) -> Self {
        Self { coord: start, dir }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_offsets() {
        for i in 0..127 {
            assert_eq!(OFFSETS[i * i], 0);
        }

        for i in 1..127 {
            assert_eq!(OFFSETS[i * i + 1] as usize, i + 1);
        }

        for i in 2..128 {
            assert_eq!(OFFSETS[i * i - 1] as usize, 3 * i * i - 4 * i + 1);
        }
    }

    #[test]
    fn test_get_round_idx() {
        for i in 2..127 {
            let i1 = i as usize - 1;
            let offsets = get_offsets(i as usize);
            assert_eq!(get_round_idx(i as u8, (-i + 1, 0).into(), offsets), 0);
            assert_eq!(get_round_idx(i as u8, (-i + 1, i - 1).into(), offsets), i1);
            assert_eq!(
                get_round_idx(i as u8, (0, 0).into(), offsets),
                3 * i1 * i as usize / 2
            );
            assert_eq!(
                get_round_idx(i as u8, (i - 1, 0).into(), offsets),
                3 * i1 * i as usize
            );
        }
    }

    #[test]
    fn test_get() {
        let b: RoundHexBoard<i32> = RoundHexBoard::new(5);
        assert_eq!(b.get((2, 3)), None);
        assert_eq!(b.get((1, 3)), Some(&0));
    }

    #[test]
    fn test_mut() {
        let mut b: RoundHexBoard<i32> = RoundHexBoard::new(5);
        let z = (0, 0);
        b[z] = 6;
        assert_eq!(b[z], 6);

        let mut b2: super::RoundHexBoard<(i32, [i32; 2])> = super::RoundHexBoard::new(5);

        b2[z].1[1] = 5;
        assert_eq!(b2[z], (0, [0, 5]));
    }
}
