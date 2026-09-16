//! Minimum-cost linear assignment.
//!
//! This is a real combinatorial solver (a shortest-augmenting-path
//! implementation of the Hungarian / Kuhn–Munkres algorithm, `O(n^3)`), not a
//! greedy nearest-neighbour pass.
//!
//! "Unmatched" is represented *inside* the problem: the cost matrix is padded
//! with one dummy column per row and one dummy row per column, weighted by
//! explicit unmatched costs. Forbidden pairs are encoded as non-finite costs
//! and are replaced internally by a large finite sentinel that is guaranteed
//! to be worse than every feasible alternative, so a forbidden pair can never
//! be selected and can never block a valid one.

use crate::error::NumericError;

/// Result of a linear assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    /// Pairs `(row, column)` that were matched.
    pub pairs: Vec<(usize, usize)>,
    /// Row indices left unmatched.
    pub unmatched_rows: Vec<usize>,
    /// Column indices left unmatched.
    pub unmatched_columns: Vec<usize>,
}

/// Solves a rectangular linear assignment problem with explicit unmatched
/// costs.
///
/// * `costs` is a `rows x num_columns` matrix. Lower is better. Any
///   non-finite entry (`NaN` or `+/-infinity`) marks a forbidden pair.
/// * `num_columns` is passed explicitly so that a problem with zero rows can
///   still report its columns as unmatched.
/// * `unmatched_row_cost` is the cost of leaving a row unmatched.
/// * `unmatched_column_cost` is the cost of leaving a column unmatched.
///
/// Both unmatched costs must be finite. A match `(i, j)` with cost `c` is
/// chosen over leaving both unmatched when `c < unmatched_row_cost +
/// unmatched_column_cost` (taking the rest of the assignment as fixed).
///
/// Returns [`NumericError`] only for non-finite unmatched costs or inconsistent
/// dimensions.
/// The assignment decomposes exactly over the connected components of the
/// "allowed pair" bipartite graph: a pair is allowed when its cost is finite.
/// There are no edges between components, so the globally optimal assignment is
/// the union of the per-component optima. This makes the common case of many
/// well-separated objects near-linear instead of cubic.
pub fn linear_assignment(
    costs: &[Vec<f64>],
    num_columns: usize,
    unmatched_row_cost: f64,
    unmatched_column_cost: f64,
) -> Result<Assignment, NumericError> {
    if !unmatched_row_cost.is_finite() || !unmatched_column_cost.is_finite() {
        return Err(NumericError::NonFiniteResult {
            context: "unmatched assignment cost",
        });
    }

    let n_rows = costs.len();
    let n_cols = num_columns;
    if costs.iter().any(|row| row.len() != n_cols) {
        return Err(NumericError::AssignmentFailed);
    }
    if n_rows == 0 || n_cols == 0 {
        return Ok(Assignment {
            pairs: Vec::new(),
            unmatched_rows: (0..n_rows).collect(),
            unmatched_columns: (0..n_cols).collect(),
        });
    }

    // Largest finite cost (including unmatched costs) determines the sentinel.
    let mut max_finite = unmatched_row_cost.max(unmatched_column_cost);
    for row in costs {
        for &c in row {
            if c.is_finite() {
                max_finite = max_finite.max(c);
            }
        }
    }
    let sentinel = max_finite + max_finite.abs().max(1.0) * 1.0e6 + 1.0e9;

    // Union rows and columns that share at least one allowed pair.
    let total = n_rows + n_cols;
    let mut union_find = UnionFind::new(total);
    for (i, row) in costs.iter().enumerate() {
        for (j, &cost) in row.iter().enumerate() {
            if cost.is_finite() {
                union_find.union(i, n_rows + j);
            }
        }
    }

    let mut component_rows: Vec<Vec<usize>> = vec![Vec::new(); total];
    let mut component_cols: Vec<Vec<usize>> = vec![Vec::new(); total];
    for i in 0..n_rows {
        let root = union_find.find(i);
        component_rows[root].push(i);
    }
    for j in 0..n_cols {
        let root = union_find.find(n_rows + j);
        component_cols[root].push(j);
    }

    let mut pairs = Vec::new();
    let mut unmatched_rows = Vec::new();
    let mut unmatched_columns = Vec::new();

    for root in 0..total {
        let rows = &component_rows[root];
        let cols = &component_cols[root];
        if rows.is_empty() && cols.is_empty() {
            continue;
        }
        if rows.is_empty() {
            unmatched_columns.extend_from_slice(cols);
            continue;
        }
        if cols.is_empty() {
            unmatched_rows.extend_from_slice(rows);
            continue;
        }

        let sub_costs: Vec<Vec<f64>> = rows
            .iter()
            .map(|&i| cols.iter().map(|&j| costs[i][j]).collect())
            .collect();
        let local = solve_dense(
            &sub_costs,
            sentinel,
            unmatched_row_cost,
            unmatched_column_cost,
        )?;
        for (local_row, local_col) in local.pairs {
            pairs.push((rows[local_row], cols[local_col]));
        }
        for local_row in local.unmatched_rows {
            unmatched_rows.push(rows[local_row]);
        }
        for local_col in local.unmatched_columns {
            unmatched_columns.push(cols[local_col]);
        }
    }

    pairs.sort_unstable();
    unmatched_rows.sort_unstable();
    unmatched_columns.sort_unstable();

    Ok(Assignment {
        pairs,
        unmatched_rows,
        unmatched_columns,
    })
}

/// Solves the dense assignment for one connected component: a set of rows and a
/// set of columns, some pairs forbidden (non-finite cost), with explicit
/// unmatched costs. Returns local indices.
fn solve_dense(
    costs: &[Vec<f64>],
    sentinel: f64,
    unmatched_row_cost: f64,
    unmatched_column_cost: f64,
) -> Result<Assignment, NumericError> {
    let n_rows = costs.len();
    let n_cols = costs.first().map_or(0, Vec::len);
    let n = n_rows + n_cols;
    // 1-indexed (n+1) x (n+1) cost matrix; index 0 is unused padding.
    let mut a = vec![vec![sentinel; n + 1]; n + 1];
    for i in 0..n_rows {
        for j in 0..n_cols {
            let c = costs[i][j];
            a[i + 1][j + 1] = if c.is_finite() { c } else { sentinel };
        }
    }
    for i in 0..n_rows {
        a[i + 1][n_cols + 1 + i] = unmatched_row_cost;
    }
    for j in 0..n_cols {
        a[n_rows + 1 + j][j + 1] = unmatched_column_cost;
    }
    // Dummy rows (unmatched columns) and dummy columns (unmatched rows) may
    // pair with each other at zero cost: when real row `i` is matched to real
    // column `j`, row `i`'s dummy column and column `j`'s dummy row are freed
    // and must absorb each other.
    for i in 0..n_cols {
        for j in 0..n_rows {
            a[n_rows + 1 + i][n_cols + 1 + j] = 0.0;
        }
    }

    let assignment = hungarian_square(&a, n)?;

    let mut pairs = Vec::new();
    let mut row_matched = vec![false; n_rows];
    let mut col_matched = vec![false; n_cols];
    // `assignment[j]` is the 1-indexed row assigned to 1-indexed column j.
    for j in 1..=n_cols {
        let row = assignment[j];
        if row >= 1 && row <= n_rows {
            pairs.push((row - 1, j - 1));
            row_matched[row - 1] = true;
            col_matched[j - 1] = true;
        }
    }

    Ok(Assignment {
        pairs,
        unmatched_rows: (0..n_rows).filter(|&i| !row_matched[i]).collect(),
        unmatched_columns: (0..n_cols).filter(|&j| !col_matched[j]).collect(),
    })
}

/// Disjoint-set forest with path halving and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            core::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}

/// Hungarian algorithm on a complete square matrix.
///
/// `a` must be `(n+1) x (n+1)` with 1-indexed rows/columns; `a[0][*]` and
/// `a[*][0]` are ignored. Returns `assignment[col] = row` (1-indexed) for each
/// column `1..=n`. All entries must be finite.
fn hungarian_square(a: &[Vec<f64>], n: usize) -> Result<Vec<usize>, NumericError> {
    if n == 0 {
        return Ok(vec![0; 1]);
    }
    let inf = f64::INFINITY;
    let mut u = vec![0.0_f64; n + 1];
    let mut v = vec![0.0_f64; n + 1];
    let mut p = vec![0_usize; n + 1];
    let mut way = vec![0_usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0_usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0_usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = a[i0][j] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            if !delta.is_finite() {
                return Err(NumericError::AssignmentFailed);
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_cost(costs: &[Vec<f64>], assignment: &Assignment) -> f64 {
        assignment.pairs.iter().map(|&(i, j)| costs[i][j]).sum()
    }

    /// Brute-force minimum over all injective row->col assignments, including
    /// leaving rows/columns unmatched.
    fn brute_force(costs: &[Vec<f64>], unmatched_row_cost: f64, unmatched_column_cost: f64) -> f64 {
        let n_rows = costs.len();
        let n_cols = costs.first().map_or(0, Vec::len);
        let mut best = f64::INFINITY;
        // Assign each row to either "unmatched" (n_cols) or a column, with
        // columns used at most once.
        let mut used_cols = vec![false; n_cols];
        #[allow(clippy::too_many_arguments)]
        fn recurse(
            row: usize,
            n_rows: usize,
            n_cols: usize,
            costs: &[Vec<f64>],
            used_cols: &mut Vec<bool>,
            current: f64,
            best: &mut f64,
            unmatched_row_cost: f64,
            unmatched_column_cost: f64,
        ) {
            if current >= *best {
                return;
            }
            if row == n_rows {
                // Remaining unused columns are all unmatched.
                let unmatched_cols = used_cols.iter().filter(|&&u| !u).count();
                let total = current + unmatched_cols as f64 * unmatched_column_cost;
                *best = best.min(total);
                return;
            }
            // Leave row unmatched.
            recurse(
                row + 1,
                n_rows,
                n_cols,
                costs,
                used_cols,
                current + unmatched_row_cost,
                best,
                unmatched_row_cost,
                unmatched_column_cost,
            );
            for j in 0..n_cols {
                if !used_cols[j] && costs[row][j].is_finite() {
                    used_cols[j] = true;
                    recurse(
                        row + 1,
                        n_rows,
                        n_cols,
                        costs,
                        used_cols,
                        current + costs[row][j],
                        best,
                        unmatched_row_cost,
                        unmatched_column_cost,
                    );
                    used_cols[j] = false;
                }
            }
        }
        recurse(
            0,
            n_rows,
            n_cols,
            costs,
            &mut used_cols,
            0.0,
            &mut best,
            unmatched_row_cost,
            unmatched_column_cost,
        );
        best
    }

    #[test]
    fn optimal_on_small_problem() {
        let costs = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let result = linear_assignment(&costs, 3, 10.0, 10.0).unwrap();
        // The optimal matching is (0,1), (1,0), (2,2) with cost 1+2+2 = 5.
        assert_eq!(total_cost(&costs, &result), 5.0);
    }

    #[test]
    fn matches_brute_force_random_square() {
        // Deterministic pseudo-random small problems.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        for n in 1..=4 {
            for _ in 0..20 {
                let costs: Vec<Vec<f64>> = (0..n)
                    .map(|_| (0..n).map(|_| next() * 10.0).collect())
                    .collect();
                let result = linear_assignment(&costs, n, 3.0, 3.0).unwrap();
                let brute = brute_force(&costs, 3.0, 3.0);
                let got = total_cost(&costs, &result)
                    + result.unmatched_rows.len() as f64 * 3.0
                    + result.unmatched_columns.len() as f64 * 3.0;
                assert!((got - brute).abs() < 1e-9, "n={n} got={got} brute={brute}");
            }
        }
    }

    #[test]
    fn rectangular_taller_and_wider() {
        let costs = vec![vec![1.0, 2.0, 3.0], vec![3.0, 1.0, 2.0]];
        let result = linear_assignment(&costs, 3, 100.0, 100.0).unwrap();
        assert_eq!(result.pairs.len(), 2);
        assert!(result.unmatched_columns.len() == 1);

        let costs = vec![vec![1.0, 2.0], vec![3.0, 1.0], vec![2.0, 4.0]];
        let result = linear_assignment(&costs, 2, 100.0, 100.0).unwrap();
        assert_eq!(result.pairs.len(), 2);
        assert_eq!(result.unmatched_rows.len(), 1);
    }

    #[test]
    fn empty_problem() {
        let result = linear_assignment(&[], 0, 1.0, 1.0).unwrap();
        assert!(result.pairs.is_empty());
        assert!(result.unmatched_rows.is_empty());

        let result = linear_assignment(&[vec![], vec![]], 0, 1.0, 1.0).unwrap();
        assert!(result.pairs.is_empty());
        assert_eq!(result.unmatched_rows, vec![0, 1]);
    }

    #[test]
    fn all_forbidden_leaves_everything_unmatched() {
        let costs = vec![
            vec![f64::INFINITY, f64::NAN],
            vec![f64::NEG_INFINITY, f64::INFINITY],
        ];
        let result = linear_assignment(&costs, 2, 5.0, 5.0).unwrap();
        assert!(result.pairs.is_empty());
        assert_eq!(result.unmatched_rows, vec![0, 1]);
        assert_eq!(result.unmatched_columns, vec![0, 1]);
    }

    #[test]
    fn forbidden_pair_does_not_block_valid_pair() {
        let costs = vec![vec![f64::INFINITY, 1.0], vec![1.0, f64::INFINITY]];
        let result = linear_assignment(&costs, 2, 100.0, 100.0).unwrap();
        assert_eq!(result.pairs.len(), 2);
        assert!(result.pairs.contains(&(0, 1)));
        assert!(result.pairs.contains(&(1, 0)));
    }

    #[test]
    fn unmatched_cost_keeps_expensive_pair_unmatched() {
        // Pair cost 8.0 exceeds unmatched alternative 2.0 + 2.0.
        let costs = vec![vec![8.0]];
        let result = linear_assignment(&costs, 1, 2.0, 2.0).unwrap();
        assert!(result.pairs.is_empty());
        assert_eq!(result.unmatched_rows, vec![0]);
        assert_eq!(result.unmatched_columns, vec![0]);

        // Pair cost 3.0 is cheaper than leaving both unmatched (4.0).
        let costs = vec![vec![3.0]];
        let result = linear_assignment(&costs, 1, 2.0, 2.0).unwrap();
        assert_eq!(result.pairs, vec![(0, 0)]);
    }

    #[test]
    fn non_finite_unmatched_cost_is_rejected() {
        let costs = vec![vec![1.0]];
        assert!(linear_assignment(&costs, 1, f64::NAN, 1.0).is_err());
    }

    #[test]
    fn tie_break_is_deterministic() {
        let costs = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let first = linear_assignment(&costs, 2, 10.0, 10.0).unwrap();
        let second = linear_assignment(&costs, 2, 10.0, 10.0).unwrap();
        assert_eq!(first, second);
    }
}
