use std::time::Instant;

use hexagonal::*;

use rand::prelude::*;

fn main() {
    let mut rng = thread_rng();
    for size in 6..13 {
        let now = Instant::now();
        let mut moves = 0;
        let mut nfields = 0;

        for _ in 0..1000 {
            let mut twgame: Tumbleweed = Tumbleweed::new(size);
            nfields = twgame.board().num_fields();
            let setup = GenStartMoves::new(twgame.board().get_coords())
                .choose(&mut rng)
                .unwrap();
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

        println!(
            "{size}: {nfields} fields, {} moves, {} us",
            moves / 1000,
            now.elapsed().as_millis()
        );
    }
}
