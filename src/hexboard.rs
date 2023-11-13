pub use hex2d::{Coordinate, Direction};
use std::hash::Hash;
use std::ops::{Index, IndexMut};

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
// Finally a `get_round_idx` translates (x, y) coordinate to index of a flat array by getting xth offset for board size s and
// adding y to that
static OFFSETS: [u16; 128*128] = init_offsets();

const fn init_offsets() -> [u16; 128*128] {
    let mut res = [0; 128*128];
    let mut i: i16 = 1;
    while i < 128 {
        let mut cnt = 0;
        let i_1 = i - 1;
        let start = i_1 * i_1;
        let mut a = -i_1;
        while a <= i {
            res[(start + a + i_1) as usize] = cnt;
            cnt += if a < 0 {
                2 * i + a - 1
            } else {
                2 * i - a - 1
            } as u16;
            a += 1;
        }
        i+= 1;
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

pub type IterField<'a, T> = std::slice::Iter<'a, T>;
pub type IterFieldMut<'a, T> = std::slice::IterMut<'a, T>;

pub type IterCoord<'a> = std::slice::Iter<'a, Coordinate>;

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
    idx: usize
}

impl<'a, T> Iterator for IterCoordField<'a, T> {
    type Item=(Coordinate, &'a T);

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

#[inline]
fn max_coord(index: Coordinate) -> u8 {
    index.x.abs().max(index.y.abs()).max(index.z().abs()) as u8
}

#[derive(Debug, Clone)]
pub struct RoundHexBoard<T>
    where
    T: FieldTrait
{
    size: u8,
    board: Box<[T]>
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
impl<T> RoundHexBoard<T>
where
    T: FieldTrait,
{

    /// Constructs a new instance of a given size. In all realistic scenatios
    /// size will be in roughly 6-12 range so instead of returning Result,
    /// the method will panic if size is greater than 127, which is well above
    /// any number that should be encountered.
    pub fn new(size: u8) -> RoundHexBoard<T> {
        if size > 127 {
            panic!("size must be less than 128");
        }
        let s = size as usize;
        let len = 3 * s * (s - 1) + 1;
        let board = vec![T::default(); len];

        RoundHexBoard {
            size,
            board: board.into_boxed_slice()
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
    pub fn iter_coord_fields(&self) -> IterCoordField<'_, T>{
        IterCoordField { size: self.size, fields: &self.board, idx: 0 }
    }

    // A safe way to access a field. If coordinate is not valid returns None.
    pub fn get(&self, index: Coordinate) -> Option<&T> {
        self.valid_coord(index).then(|| &self[index])
    }

    // A safe way to access a field. If coordinate is not valid returns None.
    pub fn get_mut(&mut self, index: Coordinate) -> Option<&mut T> {
        self.valid_coord(index).then(|| &mut self[index])
    }

    // A slice view of a board data.
    pub fn as_slice(&self) -> &[T] {
        &self.board
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

/// Generate successive coordinates in a given direction, from a given coordinate (inclusive).
/// This is an ifinite iterator.
pub struct DirectionIterator {
    coord: Coordinate,
    dir: Direction
}

impl Iterator for DirectionIterator {
    type Item = Coordinate;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.coord;
        match self.dir {
            Direction::YZ => self.coord.y += 1,
            Direction::XZ => self.coord.x += 1,
            Direction::XY => { self.coord.x += 1; self.coord.y -= 1 },
            Direction::ZY => self.coord.y -= 1,
            Direction::ZX => self.coord.x -= 1,
            Direction::YX => { self.coord.x -= 1; self.coord.y += 1 },
        }
        Some(ret)
    }
}

impl DirectionIterator {
    pub fn new(start: Coordinate, dir: Direction) -> Self {
        Self{coord: start, dir}
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_index() {
        let b: super::RoundHexBoard<i32> = super::RoundHexBoard::new(5);
        b[(-4, 0).into()];
        b[(-4, 4).into()];
        b[(-3, -1).into()];
        b[(-3, 3).into()];
        b[(-1, 3).into()];
        b[(0, 0).into()];
        b[(1, 2).into()];
        b[(2, 2).into()];
        b[(2, -4).into()];
        b[(4, 0).into()];
        b[(4, -4).into()];

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
