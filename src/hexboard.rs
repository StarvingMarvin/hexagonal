use crate::common::{HexagonalError, HexagonalResult};
pub use hex2d::{Coordinate, Direction};
use std::hash::Hash;
use std::ops::{Deref, DerefMut, Index, IndexMut};

pub trait FieldTrait: PartialEq + Clone + Hash + std::fmt::Debug + Default + Send + Sync {}
impl<T> FieldTrait for T where T: PartialEq + Clone + Hash + std::fmt::Debug + Default + Send + Sync {}

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
fn get_round_idx(board_size: u8, index: Coordinate) -> usize {
    let is1 = board_size as i32 - 1;
    let us1 = is1 as usize;
    let ux = (index.x + is1) as usize;
    let xneg = (index.x >> 31) & 1;
    let uy = (index.y + is1 + xneg * index.x) as usize;
    OFFSETS[us1 * us1 + ux] as usize + uy
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

pub type IterCoord<'a> = std::slice::Iter<'a, Coordinate>;

// Is there a better way to do this?
static mut COORDS: [Option<Box<[Coordinate]>>; 127] = init_coords();
const fn init_coords() -> [Option<Box<[Coordinate]>>; 127] {
    unsafe {
        const SIZE: usize = std::mem::size_of::<[Option<Box<[Coordinate]>>; 127]>();
        let ret = [0_u8; SIZE];
        std::mem::transmute::<[u8; SIZE], [Option<Box<[Coordinate]>>; 127]>(ret)
    }
}

fn get_coords(board_size: u8) -> &'static [Coordinate] {
    let s = board_size as usize;
    unsafe {
        let c = &mut COORDS[s];
        c.get_or_insert_with(|| {
            let mut v = Vec::with_capacity(3 * s * (s - 1) + 1);
            let is1 = board_size as i32 - 1;

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

pub struct IterCoordField<'a, T> {
    size: u8,
    fields: &'a [T],
    idx: usize,
}

impl<'a, T> Iterator for IterCoordField<'a, T> {
    type Item = (Coordinate, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.fields.len() {
            let ret = (get_coords(self.size)[self.idx], &self.fields[self.idx]);
            self.idx += 1;
            Some(ret)
        } else {
            None
        }
    }

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
    ncoord: [Coordinate; 6],
    idx: usize,
}

impl<'a, T> Iterator for IterNeighbours<'a, T>
where
    T: FieldTrait,
{
    type Item = (Coordinate, &'a T);

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

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(6 - self.idx))
    }
}

#[inline]
fn max_coord(index: Coordinate) -> u8 {
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
///assert_eq!(board[(1, 2).into()], 0);
///assert_eq!(board.get((3, 3).into()), None);
///board[(-2, 3).into()] = 4;
///assert_eq!(board.get((-2, 3).into()), Some(&4));
///```
#[derive(Debug, Clone)]
pub struct RoundHexBoard<T>
where
    T: FieldTrait,
{
    size: u8,
    board: Box<[T]>,
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
        }
    }

    /// Chechks if coordinate is falid for this board. Since this board type
    /// uses [cube coordinates](https://www.redblobgames.com/grids/hexagons/#coordinates-cube)
    /// with the origin at central field, coordinate is valid if it's less than
    /// `size` away along any axis: `max(|x|, |y|, |-x - y|) < size`.
    pub fn valid_coord(&self, coord: Coordinate) -> bool {
        max_coord(coord) < self.size
    }

    /// Board size, that is length in fields of a side of the hexagon that is this board.
    pub fn size(&self) -> u8 {
        self.size
    }

    /// Number of rows. The shortest distance in fields between two parallel sides of the board.
    pub fn num_rows(&self) -> usize {
        2 * self.size as usize - 1
    }

    /// Total number of fields on the board.
    pub fn num_fields(&self) -> usize {
        self.board.len()
    }

    /// An iterator over the board fields.
    pub fn iter_fields(&self) -> IterField<'_, T> {
        self.board.iter()
    }

    /// A mutable iterator over the board fields.
    pub fn iter_fields_mut(&mut self) -> IterFieldMut<'_, T> {
        self.board.iter_mut()
    }

    /// An iterator over the valid board coordinates.
    pub fn iter_coords(&self) -> IterCoord {
        get_coords(self.size).iter()
    }

    /// An iterator over the `(Coordinate, &T)` touples.
    pub fn iter_coord_fields(&self) -> IterCoordField<'_, T> {
        IterCoordField {
            size: self.size,
            fields: &self.board,
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
    pub fn iter_neighbours(&self, coord: Coordinate) -> IterNeighbours<T> {
        IterNeighbours {
            board: self,
            ncoord: coord.neighbors(),
            idx: 0,
        }
    }

    /// A safe way to access a field. If coordinate is not valid returns None.
    pub fn get(&self, index: Coordinate) -> Option<&T> {
        self.valid_coord(index).then(|| &self[index])
    }

    /// A safe way to access a field. If coordinate is not valid returns None.
    pub fn get_mut(&mut self, index: Coordinate) -> Option<&mut T> {
        self.valid_coord(index).then(|| &mut self[index])
    }

    /// A slice view of a board data.
    pub fn as_slice(&self) -> &[T] {
        &self.board
    }

    /// A mutable slice view of a board data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.board
    }
}

impl<T: FieldTrait> TryFrom<Box<[T]>> for RoundHexBoard<T> {
    type Error = HexagonalError;
    fn try_from(value: Box<[T]>) -> HexagonalResult<Self> {
        let size = SIZES.binary_search(&value.len());
        match size {
            Ok(s) => Ok(RoundHexBoard {
                size: s as u8,
                board: value,
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
    fn try_from(value: &[T]) -> HexagonalResult<Self> {
        let size = SIZES.binary_search(&value.len());
        match size {
            Ok(s) => Ok(RoundHexBoard {
                size: s as u8,
                board: value.into(),
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

    fn deref(&self) -> &Self::Target {
        &self.board
    }
}

impl<T: FieldTrait> DerefMut for RoundHexBoard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.board
    }
}

impl<T: FieldTrait> Index<Coordinate> for RoundHexBoard<T> {
    type Output = T;

    fn index(&self, index: Coordinate) -> &Self::Output {
        let idx: usize = get_round_idx(self.size, index);
        &self.board[idx]
    }
}

impl<T: FieldTrait> IndexMut<Coordinate> for RoundHexBoard<T> {
    fn index_mut(&mut self, index: Coordinate) -> &mut Self::Output {
        let idx: usize = get_round_idx(self.size, index);
        &mut self.board[idx]
    }
}

impl<T: FieldTrait> IntoIterator for RoundHexBoard<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Into::<Vec<T>>::into(self.board).into_iter()
    }
}

macro_rules! impl_from_arr {
    ($size:expr) => {
        impl<T: FieldTrait> From<[T; SIZES[$size]]> for RoundHexBoard<T> {
            fn from(value: [T; SIZES[$size]]) -> Self {
                RoundHexBoard {
                    size: $size,
                    board: Box::new(value),
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
    coord: Coordinate,
    dir: Direction,
}

impl Iterator for DirectionIterator {
    type Item = Coordinate;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.coord;
        self.coord = self.coord + self.dir;
        Some(ret)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

impl DirectionIterator {
    pub fn new(start: Coordinate, dir: Direction) -> Self {
        Self { coord: start, dir }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_offsets() {
        for i in 0..127 {
            assert_eq!(super::OFFSETS[i * i], 0);
        }

        for i in 1..127 {
            assert_eq!(super::OFFSETS[i * i + 1] as usize, i + 1);
        }

        for i in 2..128 {
            assert_eq!(super::OFFSETS[i * i - 1] as usize, 3 * i * i - 4 * i + 1);
        }
    }

    #[test]
    fn test_get_round_idx() {
        for i in 2..127 {
            let i1 = i as usize - 1;
            assert_eq!(super::get_round_idx(i as u8, (-i + 1, 0).into()), 0);
            assert_eq!(super::get_round_idx(i as u8, (-i + 1, i - 1).into()), i1);
            assert_eq!(
                super::get_round_idx(i as u8, (0, 0).into()),
                3 * i1 * i as usize / 2
            );
            assert_eq!(
                super::get_round_idx(i as u8, (i - 1, 0).into()),
                3 * i1 * i as usize
            );
        }
    }

    #[test]
    fn test_get() {
        let b: super::RoundHexBoard<i32> = super::RoundHexBoard::new(5);
        assert_eq!(b.get((2, 3).into()), None);
        assert_eq!(b.get((1, 3).into()), Some(&0));
    }

    #[test]
    fn test_mut() {
        let mut b: super::RoundHexBoard<i32> = super::RoundHexBoard::new(5);
        b[(0, 0).into()] = 6;
        assert_eq!(b[(0, 0).into()], 6);

        let mut b2: super::RoundHexBoard<(i32, [i32; 2])> = super::RoundHexBoard::new(5);
        b2[(0, 0).into()].1[1] = 5;
        assert_eq!(b2[(0, 0).into()], (0, [0, 5]));
    }
}
