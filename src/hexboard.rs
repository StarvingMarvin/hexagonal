pub use hex2d::Coordinate;
use std::hash::Hash;
use std::ops::{Deref, Index, IndexMut};

pub trait FieldTrait: PartialEq + Clone + Hash + std::fmt::Debug + Default + Send + Sync {}
impl<T> FieldTrait for T where T: PartialEq + Clone + Hash + std::fmt::Debug + Default + Send + Sync {}

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

pub type Iter<'a, T> = std::slice::Iter<'a, T>;
pub type IterMut<'a, T> = std::slice::IterMut<'a, T>;


#[inline]
fn get_round_idx(board_size: u8, index: Coordinate) -> usize {
    let is1 = board_size as i32 - 1;
    let us1 = is1 as usize;
    let ux = (index.x + is1) as usize;
    let xpos = (index.x >> 31) & 1;
    let s = (1 -xpos) * (index.y + is1) + xpos * (index.y + is1 + index.x);
    OFFSETS[us1 * us1 + ux] as usize + s as usize
}

#[inline]
fn max_coord(index: Coordinate) -> u8 {
    index.x.abs().max(index.x.abs()).max(index.z().abs()) as u8
}

#[derive(Debug, Clone)]
pub struct RoundHexBoard<T>
    where
    T: FieldTrait
{
    size: u8,
    board: Box<[(Coordinate, T)]>
}

impl<T> Deref for RoundHexBoard<T>
    where
    T: FieldTrait
{
    type Target = [(Coordinate, T)];

    #[inline]
    fn deref(&self) -> &[(Coordinate, T)] {
        &self.board
    }
}

impl<'a, T> RoundHexBoard<T>
where
    T: FieldTrait,
{
    pub fn new(size: u8) -> RoundHexBoard<T> {
        let s = size as usize;
        let len = 3 * s * (s - 1) + 1;
        let mut board = Vec::with_capacity(len);

        let is1 = size as i32 - 1;

        for i in -is1..=is1 {
            let (lb, ub) = if i < 0 {
                (-is1 - i, is1)
            } else {
                (-is1, is1 - i)
            };

            for j in lb..=ub {
                let c = (i, j).into();
                board.push((c, T::default()));
            }
        }

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

    pub fn iter(&self) -> Iter<'_, (Coordinate, T)> {
        self.board.iter()
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
