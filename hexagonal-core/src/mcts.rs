use rand::seq::SliceRandom;

use crate::game::*;

#[derive(Debug, Clone)]
pub struct Node<G: Game> {
    pub state: G,
    depth: u16,
    cparam: f32,
    pub n: u32,
    pub results: [u32; 3],
    children: Vec<Node<G>>,
    unvisited_children: Vec<G::Move>,
}

impl<G: Game> Node<G> {
    pub fn new(state: G, depth: u16, cparam: f32) -> Node<G> {
        let valid_moves = &state.valid_moves();
        let total_children = valid_moves.len();
        let mut rng = rand::thread_rng();
        let unvisited = valid_moves
            .choose_multiple(&mut rng, total_children)
            .cloned()
            .collect();

        Node {
            state,
            depth,
            cparam,
            n: 0,
            results: [0, 0, 0],
            children: vec![],
            unvisited_children: unvisited,
        }
    }

    pub fn visit(&mut self) -> i32 {
        self.n += 1;

        let result = if self.state.game_over()
            || (self.unvisited_children.is_empty() && self.children.is_empty())
        {
            self.state.result()
        } else {
            match self.unvisited_children.pop() {
                Some(m) => {
                    let mut g = self.state.clone();
                    let lst = self.children.len();
                    g.play(m);
                    let nxt = Node::new(g, self.depth, self.cparam);
                    self.children.push(nxt);
                    self.children[lst].rollout(self.depth)
                }
                None => self.best_child(self.cparam).visit(),
            }
        };

        self.results[((result + 2) % 3) as usize] += 1;
        result
    }

    pub fn rollout(&mut self, depth: u16) -> i32 {
        self.n += 1;
        let mut g = self.state.clone();
        let mut i = 0;
        let mut rng = rand::thread_rng();

        while let Some(m) = g.valid_moves().choose(&mut rng).copied() {
            g.play(m);
            i += 1;
            if i >= depth {
                break;
            }
        }

        let result = g.result();
        self.results[((result + 2) % 3) as usize] += 1;
        result
    }

    fn best_child(&mut self, cparam: f32) -> &mut Node<G> {
        self.children
            .iter_mut()
            .max_by_key(|n| {
                let nf = n.n as f32;
                let w = n.q() / nf + cparam * (2.0 * (self.n as f32).ln() / nf).sqrt();
                (w * 100.0) as i32
            })
            .unwrap()
    }

    pub fn q(&self) -> f32 {
        let wins = self.results[self.state.next_player().into()] as f32;
        let losses = self.results[self.state.current_player().into()] as f32;
        wins - losses
    }

    pub fn pick(&self) -> &Node<G> {
        self.children.iter().max_by_key(|&n| n.n).unwrap()
    }
}

impl<G: Game> From<&Node<G>> for AiResultNode<G::Move> {
    fn from(value: &Node<G>) -> Self {
        AiResultNode {
            q: value.q(),
            n: value.n,
            children: value
                .children
                .iter()
                .map(|n| AiResultLeaf {
                    q: n.q(),
                    n: n.n,
                    mv: n.state.last_move().unwrap(),
                })
                .collect(),
        }
    }
}

impl<G: Game> From<&Node<G>> for AiResultLeaf<G::Move> {
    fn from(value: &Node<G>) -> Self {
        AiResultLeaf {
            q: value.q(),
            n: value.n,
            mv: value.state.last_move().unwrap(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Mcts {
    pub depth: u16,
    pub cparam: f32,
    pub rollouts: u32,
    pub max_n: u32,
    pub max_q: f32,
}

impl<G: Game + 'static> for Mcts {
    fn start_state(&self, game: &G) -> AiFuture<<G as Game>::Move> {
        let (tx, rx) = mpsc::sync_channel(1);
        let gs = game.clone();
        let loc = *self;
        thread::spawn(move || {
            mcts_loop(gs, loc.depth, loc.cparam, loc.rollouts, loc.max_n, loc.max_q, tx);
        });
        AiFuture { rx }
    }
}

pub fn mcts_loop<G: AiGame>(
    start_state: G,
    depth: u16,
    cparam: f32,
    rollouts: u32,
    n: u32,
    q: f32,
    tx: SyncSender<AiResult<G::Move>>,
) {
    let mut root = Node::new(start_state, depth, cparam);
    if root.state.game_over() {
        let _ = tx.send(AiResult::GameOver);
        return;
    }

    let mut pick;

    loop {
        root.visit();
        pick = root.pick();
        if root.n > rollouts || pick.n > n || pick.q() > q || pick.q() < -q {
            break;
        }
        let _ = tx.try_send(AiResult::Partial((&root).into()));
    }
    let _ = tx.send(AiResult::Final(pick.into()));
}
