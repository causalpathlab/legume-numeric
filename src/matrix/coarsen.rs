//! Groups of the nodes of a weighted graph, by rounds of heavy-edge matching
//! (the coarsening phase of multilevel partitioners such as METIS), and the
//! graph between the groups.
//!
//! Each round visits the groups in a seeded random order and merges each
//! still-unmatched group with the unmatched neighbour it is most strongly
//! linked to, per pair of members (`w / (sᵤ·sᵥ)`, so small groups pair up
//! first and sizes stay even), if that link is at least half its strongest
//! one; otherwise it waits, and groups waiting on the same neighbour pair
//! up with each other (two-hop matching), so a hub does not hold the
//! rounds back. Rounds stop
//! at the target count. Groups are merged only through an edge or a shared
//! neighbour, so a graph of more pieces than the target keeps one group per
//! piece at least.
//!
//! Each round is a pass over the edges, so the whole is
//! `O(E log E · log(n / target))`.

use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

/// A group merges only with a neighbour linked at least this share as
/// strongly as its strongest neighbour, taken or not; otherwise it waits.
/// Without it, a group whose strong neighbours are taken settles for a
/// weak link, such as a stray edge between two separate populations.
const STRONG: f32 = 0.5;

/// The group of each of `n` nodes (ids `0..groups`) after heavy-edge
/// matching on the undirected `edges` (`(i, j, weight)`) down to `target`
/// groups, or as few as the graph allows. Ties break by `seed`.
#[must_use]
pub fn coarsen(n: usize, edges: &[(usize, usize, f32)], target: usize, seed: u64) -> Vec<usize> {
    let mut membership: Vec<usize> = (0..n).collect();
    if target >= n {
        return membership;
    }
    let mut size = vec![1_usize; n];
    let mut graph = collapse(edges, &membership);
    let mut count = n;
    let mut rng = SmallRng::seed_from_u64(seed);

    while count > target {
        let around = adjacency(count, &graph);
        let mut order: Vec<usize> = (0..count).collect();
        order.shuffle(&mut rng);

        let mut mate = vec![usize::MAX; count];
        let mut merges = 0;
        // Groups left waiting this round, by their strongest neighbour.
        let mut waiting: Vec<(usize, usize)> = Vec::new();
        for &u in &order {
            if count - merges <= target {
                break;
            }
            if mate[u] != usize::MAX {
                continue;
            }
            let link = |&(v, w): &(usize, f32)| w / (size[u] * size[v]) as f32;
            let Some(hub) = around[u].iter().max_by(|a, b| link(a).total_cmp(&link(b))) else {
                continue;
            };
            let bar = STRONG * link(hub);
            let best = around[u]
                .iter()
                .filter(|&&(v, _)| mate[v] == usize::MAX)
                .filter(|e| link(e) >= bar)
                .max_by(|a, b| link(a).total_cmp(&link(b)));
            if let Some(&(v, _)) = best {
                mate[u] = v;
                mate[v] = u;
                merges += 1;
            } else {
                waiting.push((hub.0, u));
            }
        }
        // Groups waiting on the same neighbour pair up with each other
        // (two-hop matching, as METIS does): else a hub every one of them
        // links to most takes one of them a round.
        waiting.retain(|&(_, u)| mate[u] == usize::MAX);
        waiting.sort_by_key(|&(hub, _)| hub);
        for [(_, u), (_, v)] in waiting
            .chunk_by(|a, b| a.0 == b.0)
            .flat_map(|g| g.as_chunks::<2>().0)
            .copied()
        {
            if count - merges <= target {
                break;
            }
            mate[u] = v;
            mate[v] = u;
            merges += 1;
        }
        if merges == 0 {
            break;
        }

        // Each pair, or each group left alone, becomes one new group.
        let mut next = vec![usize::MAX; count];
        let mut fresh = 0;
        for u in 0..count {
            if next[u] == usize::MAX {
                next[u] = fresh;
                if mate[u] != usize::MAX {
                    next[mate[u]] = fresh;
                }
                fresh += 1;
            }
        }
        let mut new_size = vec![0; fresh];
        for (u, &s) in size.iter().enumerate() {
            new_size[next[u]] += s;
        }
        membership.iter_mut().for_each(|g| *g = next[*g]);
        graph = collapse(&graph, &next);
        size = new_size;
        count = fresh;
    }
    membership
}

/// The graph between groups: for each pair of groups linked by any of the
/// undirected `edges`, the sum of their weights, as `(a, b, weight)` with
/// `a < b`, sorted. Edges inside a group, and edges of a node in no group
/// (`usize::MAX`), are dropped.
#[must_use]
pub fn collapse(edges: &[(usize, usize, f32)], membership: &[usize]) -> Vec<(usize, usize, f32)> {
    let mut between: Vec<(usize, usize, f32)> = edges
        .iter()
        .filter_map(|&(i, j, w)| {
            let (a, b) = (membership[i], membership[j]);
            (a != b && a != usize::MAX && b != usize::MAX).then(|| (a.min(b), a.max(b), w))
        })
        .collect();
    between.sort_by_key(|&(a, b, _)| (a, b));
    let mut out: Vec<(usize, usize, f32)> = Vec::with_capacity(between.len());
    for (a, b, w) in between {
        match out.last_mut() {
            Some(last) if (last.0, last.1) == (a, b) => last.2 += w,
            _ => out.push((a, b, w)),
        }
    }
    out
}

/// Each node's neighbours and edge weights, both ways.
fn adjacency(n: usize, edges: &[(usize, usize, f32)]) -> Vec<Vec<(usize, f32)>> {
    let mut around = vec![Vec::new(); n];
    for &(a, b, w) in edges {
        around[a].push((b, w));
        around[b].push((a, w));
    }
    around
}

#[cfg(test)]
mod tests;
