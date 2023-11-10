pub use hex2d::Coordinate;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

pub trait FieldTrait: PartialEq + Clone + Hash + std::fmt::Debug + Default + Send + Sync {}
impl<T> FieldTrait for T where T: PartialEq + Clone + Hash + std::fmt::Debug + Default + Send + Sync {}

// RoundHexBoard saves all the data into a continuous slice so it needs to resolve an (x, y) coordinate into the slice index.
// Since not every row has the same nuber of columns, this array saves starting offsets for each row for board sizes 1-127.
// Offsets can be retrieved from a flat array because of the following observations:
//
//     * number of rows for the board of (side) size is 2s - 1
//     * sum{1..n}(2s - 1) = n^2
//     * therefore offsets for row starts for board of size s is located in OFFSETS[(s - 1)^2..s^2]
//
// finally a `get_round_idx` translates (x, y) coordinate to index of a flat array by getting xth offset for board size s and
// ads y to that
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
    fields: &'a Box<[T]>,
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

impl<'a, T> RoundHexBoard<T>
where
    T: FieldTrait,
{
    pub fn new(size: u8) -> RoundHexBoard<T> {
        let s = size as usize;
        let len = 3 * s * (s - 1) + 1;
        let board = vec![T::default(); len];

        RoundHexBoard {
            size,
            board: board.into_boxed_slice()
        }
    }

    pub fn valid_coord(&self, coord: Coordinate) -> bool {
        max_coord(coord) < self.size
    }

    pub fn size(&self) -> usize {
        self.size as usize
    }

    pub fn iter_fields(&self) -> IterField<'_, T> {
        self.board.iter()
    }

    pub fn iter_fields_mut(&mut self) -> IterFieldMut<'_, T> {
        self.board.iter_mut()
    }

    pub fn iter_coords(&self) -> IterCoord {
        get_coords(self.size).iter()
    }

    pub fn iter_coord_fields(&self) -> IterCoordField<'_, T>{
        IterCoordField { size: self.size, fields: &self.board, idx: 0 }
    }

    pub fn get(&self, index: Coordinate) -> Option<&T> {
        if self.valid_coord(index) {
            Some(&self[index])
        } else {
            None
        }

    }

    pub fn get_mut(&mut self, index: Coordinate) -> Option<&mut T> {
        if self.valid_coord(index) {
            Some(&mut self[index])
        } else {
            None
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.board
    }

}

impl<'a, T: FieldTrait> Index<Coordinate> for RoundHexBoard<T> {
    type Output = T;

    fn index(&self, index: Coordinate) -> &Self::Output {
        let idx: usize = get_round_idx(self.size, index);
        assert_eq!(self.board[idx].0, index);

        &self.board[idx].1
    }
}

impl<'a, T: FieldTrait> IndexMut<Coordinate> for RoundHexBoard<T> {
    fn index_mut(&mut self, index: Coordinate) -> &mut Self::Output {
        let idx: usize = get_round_idx(self.size, index);
        assert_eq!(self.board[idx].0, index);

        &mut self.board[idx].1
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
