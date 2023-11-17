use std::time::Instant;

use hexagonal::*;

use rand::prelude::*;

fn main() {
    let mut rng = thread_rng();
    let now = Instant::now();
    let mut moves = 0;

    for _ in 0..1000 {
        let mut twgame = Tumbleweed::new(10);
        let setup = twgame.start_move_iter().choose(&mut rng).unwrap();
        twgame.play_unchecked(setup);

        while !twgame.game_over() {
            let valid = twgame.valid_moves();
            let mv: TumbleweedMove = *valid.choose(&mut rng).unwrap();
            match mv {
                TumbleweedMove::Swap | TumbleweedMove::Pass if valid.len() > 1 => {
                    continue;
                }
                _ => (),
            }
            twgame.play_unchecked(mv);
        }
        moves += twgame.played_moves().len();
    }

    println!("{moves} moves, {} ms", now.elapsed().as_millis());
}
