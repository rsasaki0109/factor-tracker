//! Fixed-lag re-association of LiDAR tracks by factor-graph cost.
//!
//! The engine associates each scan on arrival. [`FixedLagRefiner`] keeps the
//! last `lag_frames` processed scans and re-decides, over that window, which
//! detection belongs to which person, then emits each scan once it leaves the
//! window. Only the association uses the lag; positions are re-estimated
//! causally.
//!
//! Cost of a hypothesis (negative log-likelihood, lower is better):
//!
//! ```text
//! per track      birth (if it starts in the window) + trajectory cost + misses * miss_cost
//! per detection  clutter_cost if it belongs to no track
//! ```
//!
//! The trajectory cost is the optimum of the track's factor graph (constant
//! velocity motion factors and position factors, horizontal plane). For this
//! linear-Gaussian graph it equals the Kalman filter's prediction-error
//! decomposition, which is how it is computed; x and y share one covariance.
//!
//! The search starts from the engine's association and applies only moves that
//! lower the cost: swapping the window parts of two tracks after a scan (a
//! track that moved to another person), one track absorbing another's window
//! part (a duplicate), splitting a track, and dropping one detection as
//! clutter. Scans that left the window are frozen into each track's filter
//! state and never change, so emitted output is final.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::engine::{LidarAssociationOutput, LidarFrameOutput, TrackOutput};

/// Refiner configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct FgoConfig {
    /// Window length in scans; a scan is emitted this many scans late.
    pub lag_frames: usize,
    /// Two tracks are compared when their detections come this close (m).
    pub radius_m: f64,
    /// Cost of starting a track.
    pub birth_cost: f64,
    /// Cost per scan a track has no detection (between its first and last).
    pub miss_cost: f64,
    /// Cost of a detection that belongs to no track.
    pub clutter_cost: f64,
    /// Acceleration spectral density of the motion model (m^2/s^3).
    pub accel_spectral_density: f64,
    /// Detection position standard deviation (m).
    pub position_sigma_m: f64,
    /// Velocity variance of a new track (m^2/s^2).
    pub initial_velocity_variance: f64,
    /// A track is reported for this long after its last detection.
    pub coast_ns: i64,
    /// A track without detections for this long is forgotten.
    pub forget_after_ns: i64,
    /// Upper bound on local-search passes per scan.
    pub max_passes: usize,
}

impl Default for FgoConfig {
    fn default() -> Self {
        // Tuned on five CODa windows and GrandTour ETH-2 (scripts/fgo).
        FgoConfig {
            lag_frames: 10,
            radius_m: 1.5,
            birth_cost: 25.0,
            miss_cost: 0.3,
            clutter_cost: 30.0,
            accel_spectral_density: 0.5,
            position_sigma_m: 0.25,
            initial_velocity_variance: 1.0,
            coast_ns: 500_000_000,
            forget_after_ns: 2_000_000_000,
            max_passes: 20,
        }
    }
}

/// Constant-velocity filter on the horizontal plane; x and y share `p`.
#[derive(Debug, Clone, Copy)]
struct Kf {
    x: [f64; 2],
    v: [f64; 2],
    p00: f64,
    p01: f64,
    p11: f64,
    t: f64,
}

impl Kf {
    fn start(z: &[f64], t: f64, c: &FgoConfig) -> Kf {
        Kf {
            x: [z[0], z[1]],
            v: [0.0, 0.0],
            p00: c.position_sigma_m.powi(2),
            p01: 0.0,
            p11: c.initial_velocity_variance,
            t,
        }
    }

    fn predict(&mut self, t: f64, q: f64) {
        let dt = t - self.t;
        let (p00, p01, p11) = (self.p00, self.p01, self.p11);
        self.p00 = p00 + 2.0 * dt * p01 + dt * dt * p11 + q * dt.powi(3) / 3.0;
        self.p01 = p01 + dt * p11 + q * dt * dt / 2.0;
        self.p11 = p11 + q * dt;
        for i in 0..2 {
            self.x[i] += dt * self.v[i];
        }
        self.t = t;
    }

    /// Updates with a position and returns the observation's NLL.
    fn update(&mut self, z: &[f64], r: f64) -> f64 {
        let s = self.p00 + r;
        let (k0, k1) = (self.p00 / s, self.p01 / s);
        let mut nll = 0.0;
        for ((x, v), z) in self.x.iter_mut().zip(self.v.iter_mut()).zip(z) {
            let nu = z - *x;
            nll += 0.5 * (nu * nu / s + (2.0 * std::f64::consts::PI * s).ln());
            *x += k0 * nu;
            *v += k1 * nu;
        }
        let (p00, p01, p11) = (self.p00, self.p01, self.p11);
        self.p00 = (1.0 - k0) * p00;
        self.p01 = (1.0 - k0) * p01;
        self.p11 = p11 - k1 * p01;
        nll
    }
}

/// The frozen part of a track: its filter after its last frozen detection.
#[derive(Debug, Clone)]
struct Prefix {
    kf: Kf,
    last_frame: usize,
    last_ns: i64,
    count: u32,
    heights: VecDeque<f64>,
    class_label: Option<u32>,
}

/// A detection: (scan index, observation index in that scan).
type Det = (usize, usize);

/// A move on two tracks: their new window parts, detections that become
/// clutter, and the kind of move.
type PairMove = (Vec<Det>, Vec<Det>, Vec<Det>, &'static str);

/// A move on one track: its new window part, a split-off tail, a dropped
/// detection.
type TrackMove = (Vec<Det>, Option<Vec<Det>>, Option<Det>);

#[derive(Debug, Clone, Default)]
struct Track {
    prefix: Option<Prefix>,
    window: Vec<Det>,
}

struct Scan {
    output: LidarFrameOutput,
    t: f64,
}

/// Counters of applied moves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FgoCounters {
    /// Window parts of two tracks exchanged.
    pub swaps: u64,
    /// One track absorbed another's window part.
    pub merges: u64,
    /// A track split in two.
    pub splits: u64,
    /// Detections declared clutter.
    pub drops: u64,
}

/// Fixed-lag re-association of engine LiDAR frames. See the module docs.
pub struct FixedLagRefiner {
    config: FgoConfig,
    scans: VecDeque<Scan>,
    first: usize,
    tracks: BTreeMap<u64, Track>,
    clutter: BTreeSet<Det>,
    engine_owner: HashMap<u64, u64>,
    engine_of: HashMap<Det, u64>,
    next_id: u64,
    counters: FgoCounters,
}

impl FixedLagRefiner {
    /// Creates a refiner.
    pub fn new(config: FgoConfig) -> Self {
        FixedLagRefiner {
            config,
            scans: VecDeque::new(),
            first: 0,
            tracks: BTreeMap::new(),
            clutter: BTreeSet::new(),
            engine_owner: HashMap::new(),
            engine_of: HashMap::new(),
            next_id: 1,
            counters: FgoCounters::default(),
        }
    }

    /// Applied moves so far.
    pub fn counters(&self) -> FgoCounters {
        self.counters
    }

    /// Adds the engine's next LiDAR frame; returns the frames that left the
    /// window (oldest first), with refined track ids and states.
    pub fn push(&mut self, frame: LidarFrameOutput) -> Vec<LidarFrameOutput> {
        let g = self.first + self.scans.len();
        let t = frame.timestamp_ns as f64 * 1e-9;
        for association in &frame.associations {
            let det = (g, association.observation_index);
            let Some(engine_id) = association.track_id else {
                self.clutter.insert(det);
                continue;
            };
            self.engine_of.insert(det, engine_id);
            let owner = self.engine_owner.get(&engine_id).copied().filter(|id| {
                self.tracks
                    .get(id)
                    .is_some_and(|track| track.window.last().is_none_or(|d| d.0 != g))
            });
            let id = owner.unwrap_or_else(|| {
                let id = self.next_id;
                self.next_id += 1;
                id
            });
            self.engine_owner.insert(engine_id, id);
            self.tracks.entry(id).or_default().window.push(det);
        }
        self.scans.push_back(Scan { output: frame, t });
        self.improve();
        let mut out = Vec::new();
        while self.scans.len() > self.config.lag_frames {
            out.push(self.freeze_oldest());
        }
        out
    }

    /// Emits every frame still in the window.
    pub fn finish(&mut self) -> Vec<LidarFrameOutput> {
        let mut out = Vec::new();
        while !self.scans.is_empty() {
            out.push(self.freeze_oldest());
        }
        out
    }

    fn position(&self, det: Det) -> &[f64] {
        &self.scans[det.0 - self.first].output.observations[det.1]
    }

    fn time(&self, frame: usize) -> f64 {
        self.scans[frame - self.first].t
    }

    /// Cost of a track's window part given its frozen prefix.
    fn cost(&self, prefix: Option<&Prefix>, window: &[Det]) -> f64 {
        let c = &self.config;
        let r = c.position_sigma_m.powi(2);
        let Some(first) = window.first() else {
            return 0.0;
        };
        let (mut kf, mut last, mut total, rest) = match prefix {
            Some(prefix) => (prefix.kf, prefix.last_frame, 0.0, window),
            None => (
                Kf::start(self.position(*first), self.time(first.0), c),
                first.0,
                c.birth_cost,
                &window[1..],
            ),
        };
        let mut misses = 0usize;
        for det in rest {
            misses += det.0 - last - 1;
            kf.predict(self.time(det.0), c.accel_spectral_density);
            total += kf.update(self.position(*det), r);
            last = det.0;
        }
        total + misses as f64 * c.miss_cost
    }

    fn track_cost(&self, id: u64) -> f64 {
        let track = &self.tracks[&id];
        self.cost(track.prefix.as_ref(), &track.window)
    }

    /// Pairs of tracks whose window detections come within `radius_m` in
    /// one scan or consecutive scans, with the scans where they do.
    fn neighbours(&self) -> BTreeMap<(u64, u64), BTreeSet<usize>> {
        let mut by_scan: BTreeMap<usize, Vec<(u64, [f64; 2])>> = BTreeMap::new();
        for (&id, track) in &self.tracks {
            for &det in &track.window {
                let p = self.position(det);
                by_scan.entry(det.0).or_default().push((id, [p[0], p[1]]));
            }
        }
        let r2 = self.config.radius_m.powi(2);
        let mut pairs: BTreeMap<(u64, u64), BTreeSet<usize>> = BTreeMap::new();
        for (&frame, entries) in &by_scan {
            let next = by_scan.get(&(frame + 1));
            for (a, pa) in entries {
                for (b, pb) in entries.iter().chain(next.into_iter().flatten()) {
                    if a != b && (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) < r2 {
                        pairs
                            .entry((*a.min(b), *a.max(b)))
                            .or_default()
                            .insert(frame);
                    }
                }
            }
        }
        pairs
    }

    fn improve(&mut self) {
        let clutter = self.config.clutter_cost;
        for _ in 0..self.config.max_passes {
            let mut changed = false;
            for ((a, b), frames) in self.neighbours() {
                if !self.tracks.contains_key(&a) || !self.tracks.contains_key(&b) {
                    continue;
                }
                let (ta, tb) = (&self.tracks[&a], &self.tracks[&b]);
                let base = self.track_cost(a) + self.track_cost(b);
                let mut best: (f64, Option<PairMove>) = (0.0, None);
                let cuts: BTreeSet<usize> = frames.iter().flat_map(|&f| [f, f + 1]).collect();
                for f in cuts {
                    let na: Vec<Det> = ta
                        .window
                        .iter()
                        .filter(|d| d.0 < f)
                        .chain(tb.window.iter().filter(|d| d.0 >= f))
                        .copied()
                        .collect();
                    let nb: Vec<Det> = tb
                        .window
                        .iter()
                        .filter(|d| d.0 < f)
                        .chain(ta.window.iter().filter(|d| d.0 >= f))
                        .copied()
                        .collect();
                    if na == ta.window {
                        continue;
                    }
                    let d = self.cost(ta.prefix.as_ref(), &na) + self.cost(tb.prefix.as_ref(), &nb)
                        - base;
                    if d < best.0 - 1e-9 {
                        best = (d, Some((na, nb, Vec::new(), "swap")));
                    }
                }
                // One track absorbs the other's window part; where both have a
                // detection in a scan, one is kept and the other is clutter.
                for (keeper, other) in [(ta, tb), (tb, ta)] {
                    for keep_own in [true, false] {
                        let mut by_frame: BTreeMap<usize, Det> = BTreeMap::new();
                        let mut dropped = Vec::new();
                        for &det in &keeper.window {
                            by_frame.insert(det.0, det);
                        }
                        for &det in &other.window {
                            match by_frame.get(&det.0) {
                                Some(_) if keep_own => dropped.push(det),
                                Some(&own) => {
                                    dropped.push(own);
                                    by_frame.insert(det.0, det);
                                }
                                None => {
                                    by_frame.insert(det.0, det);
                                }
                            }
                        }
                        let merged: Vec<Det> = by_frame.into_values().collect();
                        let d = self.cost(keeper.prefix.as_ref(), &merged)
                            + dropped.len() as f64 * clutter
                            - base;
                        if d < best.0 - 1e-9 {
                            let (na, nb) = if std::ptr::eq(keeper, ta) {
                                (merged, Vec::new())
                            } else {
                                (Vec::new(), merged)
                            };
                            best = (d, Some((na, nb, dropped, "merge")));
                        }
                    }
                }
                let Some((na, nb, dropped, kind)) = best.1 else {
                    continue;
                };
                self.tracks.get_mut(&a).expect("track").window = na;
                self.tracks.get_mut(&b).expect("track").window = nb;
                self.clutter.extend(dropped);
                match kind {
                    "swap" => self.counters.swaps += 1,
                    _ => self.counters.merges += 1,
                }
                changed = true;
            }
            for id in self.tracks.keys().copied().collect::<Vec<_>>() {
                let track = &self.tracks[&id];
                let base = self.track_cost(id);
                let mut best: (f64, Option<TrackMove>) = (0.0, None);
                for j in 1..track.window.len() {
                    let d = self.cost(track.prefix.as_ref(), &track.window[..j])
                        + self.cost(None, &track.window[j..])
                        - base;
                    if d < best.0 - 1e-9 {
                        best = (
                            d,
                            Some((
                                track.window[..j].to_vec(),
                                Some(track.window[j..].to_vec()),
                                None,
                            )),
                        );
                    }
                }
                let detections = track.window.len() + usize::from(track.prefix.is_some());
                if detections > 1 {
                    for j in 0..track.window.len() {
                        let mut rest = track.window.clone();
                        let det = rest.remove(j);
                        let d = self.cost(track.prefix.as_ref(), &rest) + clutter - base;
                        if d < best.0 - 1e-9 {
                            best = (d, Some((rest, None, Some(det))));
                        }
                    }
                }
                let Some((keep, split, drop)) = best.1 else {
                    continue;
                };
                self.tracks.get_mut(&id).expect("track").window = keep;
                if let Some(tail) = split {
                    let new = self.next_id;
                    self.next_id += 1;
                    self.tracks.insert(
                        new,
                        Track {
                            prefix: None,
                            window: tail,
                        },
                    );
                    self.counters.splits += 1;
                }
                if let Some(det) = drop {
                    self.clutter.insert(det);
                    self.counters.drops += 1;
                }
                changed = true;
            }
            self.tracks
                .retain(|_, track| track.prefix.is_some() || !track.window.is_empty());
            if !changed {
                break;
            }
        }
        // The engine's next detection for one of its tracks continues the
        // refined track that now holds that engine track's latest detection.
        let mut latest: HashMap<u64, (usize, u64)> = HashMap::new();
        for (&id, track) in &self.tracks {
            for det in &track.window {
                if let Some(&engine_id) = self.engine_of.get(det) {
                    let entry = latest.entry(engine_id).or_insert((det.0, id));
                    if det.0 >= entry.0 {
                        *entry = (det.0, id);
                    }
                }
            }
        }
        for (engine_id, (_, id)) in latest {
            self.engine_owner.insert(engine_id, id);
        }
    }

    fn freeze_oldest(&mut self) -> LidarFrameOutput {
        let g = self.first;
        let c = self.config.clone();
        let r = c.position_sigma_m.powi(2);
        let t = self.time(g);
        let scan_ns = self.scans[0].output.timestamp_ns;
        let engine_class: HashMap<usize, Option<u32>> = {
            let output = &self.scans[0].output;
            let class_of: HashMap<u64, Option<u32>> = output
                .tracks
                .iter()
                .map(|t| (t.track_id, t.class_label))
                .collect();
            output
                .associations
                .iter()
                .filter_map(|a| {
                    a.track_id
                        .map(|id| (a.observation_index, class_of.get(&id).copied().flatten()))
                })
                .collect()
        };
        let mut owner: HashMap<usize, u64> = HashMap::new();
        let ids: Vec<u64> = self.tracks.keys().copied().collect();
        for id in &ids {
            let det = match self.tracks[id].window.first() {
                Some(&det) if det.0 == g => det,
                _ => continue,
            };
            let z = self.position(det).to_vec();
            let class_label = engine_class.get(&det.1).copied().flatten();
            let track = self.tracks.get_mut(id).expect("track");
            track.window.remove(0);
            match &mut track.prefix {
                Some(prefix) => {
                    prefix.kf.predict(t, c.accel_spectral_density);
                    prefix.kf.update(&z, r);
                    prefix.count += 1;
                    prefix.last_frame = g;
                    prefix.last_ns = scan_ns;
                    prefix.heights.push_back(z[2]);
                    if prefix.heights.len() > 5 {
                        prefix.heights.pop_front();
                    }
                    prefix.class_label = class_label.or(prefix.class_label);
                }
                None => {
                    track.prefix = Some(Prefix {
                        kf: Kf::start(&z, t, &c),
                        last_frame: g,
                        last_ns: scan_ns,
                        count: 1,
                        heights: VecDeque::from([z[2]]),
                        class_label,
                    });
                }
            }
            owner.insert(det.1, *id);
        }
        let sensors: Vec<u32> = self.scans[0]
            .output
            .tracks
            .first()
            .map(|t| t.contributing_sensors.clone())
            .unwrap_or_default();
        let mut tracks = Vec::new();
        for (&id, track) in &self.tracks {
            let Some(prefix) = &track.prefix else {
                continue;
            };
            let age = scan_ns - prefix.last_ns;
            if age > c.coast_ns {
                continue;
            }
            let mut kf = prefix.kf;
            kf.predict(t, c.accel_spectral_density);
            let mut covariance = vec![0.0; 36];
            for (i, j, value) in [
                (0, 0, kf.p00),
                (1, 1, kf.p00),
                (2, 2, r),
                (3, 3, kf.p11),
                (4, 4, kf.p11),
                (5, 5, r),
                (0, 3, kf.p01),
                (3, 0, kf.p01),
                (1, 4, kf.p01),
                (4, 1, kf.p01),
            ] {
                covariance[6 * i + j] = value;
            }
            let height = prefix.heights.iter().sum::<f64>() / prefix.heights.len() as f64;
            tracks.push(TrackOutput {
                track_id: id,
                lifecycle: if prefix.count >= 2 {
                    "confirmed"
                } else {
                    "tentative"
                }
                .to_string(),
                state_timestamp_ns: scan_ns,
                last_spatial_update_ns: Some(prefix.last_ns),
                last_camera_support_ns: None,
                age_since_spatial_update_ns: age,
                prediction_only: age > 0,
                position: vec![kf.x[0], kf.x[1], height],
                velocity: vec![kf.v[0], kf.v[1], 0.0],
                covariance,
                class_label: prefix.class_label,
                class_prior: None,
                contributing_sensors: sensors.clone(),
                observation_times_count: prefix.count,
                confidence: None,
            });
        }
        let forget = c.forget_after_ns;
        self.tracks.retain(|_, track| {
            !track.window.is_empty()
                || track
                    .prefix
                    .as_ref()
                    .is_some_and(|p| scan_ns - p.last_ns <= forget)
        });
        self.clutter.retain(|det| det.0 != g);
        self.engine_of.retain(|det, _| det.0 != g);
        let scan = self.scans.pop_front().expect("scan");
        self.first += 1;
        let mut output = scan.output;
        output.associations = output
            .associations
            .iter()
            .map(|a| {
                let id = owner.get(&a.observation_index).copied();
                LidarAssociationOutput {
                    observation_index: a.observation_index,
                    track_id: id,
                    outcome: if id.is_some() {
                        a.outcome.clone()
                    } else {
                        "unmatched".to_string()
                    },
                    cost: a.cost,
                }
            })
            .collect();
        output.unmatched_track_ids = tracks
            .iter()
            .filter(|t| t.prediction_only)
            .map(|t| t.track_id)
            .collect();
        output.tracks = tracks;
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TrackingMode;

    /// Two walkers; `engine_id(frame, walker)` is the engine's track for
    /// each walker's detection.
    fn frames(
        start: [[f64; 2]; 2],
        velocity: [[f64; 2]; 2],
        n: usize,
        engine_id: impl Fn(usize, usize) -> u64,
    ) -> Vec<LidarFrameOutput> {
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.1;
                let observations = (0..2)
                    .map(|w| {
                        vec![
                            start[w][0] + velocity[w][0] * t,
                            start[w][1] + velocity[w][1] * t,
                            1.0,
                        ]
                    })
                    .collect();
                let associations = (0..2)
                    .map(|w| LidarAssociationOutput {
                        observation_index: w,
                        track_id: Some(engine_id(i, w)),
                        outcome: "updated".to_string(),
                        cost: None,
                    })
                    .collect();
                LidarFrameOutput {
                    mode: TrackingMode::LidarOnly,
                    timestamp_ns: (i as i64) * 100_000_000,
                    tracks: Vec::new(),
                    observations,
                    associations,
                    unmatched_track_ids: Vec::new(),
                    pose_correction: None,
                    smoothed: false,
                }
            })
            .collect()
    }

    fn run(input: Vec<LidarFrameOutput>) -> (Vec<LidarFrameOutput>, FgoCounters) {
        let mut refiner = FixedLagRefiner::new(FgoConfig::default());
        let mut out = Vec::new();
        for frame in input {
            out.extend(refiner.push(frame));
        }
        out.extend(refiner.finish());
        (out, refiner.counters())
    }

    fn owner(frame: &LidarFrameOutput, observation: usize) -> Option<u64> {
        frame.associations[observation].track_id
    }

    #[test]
    fn separate_walkers_pass_through() {
        let input = frames(
            [[0.0, 0.0], [0.0, 5.0]],
            [[1.0, 0.0], [-1.0, 0.0]],
            30,
            |_, w| w as u64 + 7,
        );
        let (out, counters) = run(input);
        assert_eq!(out.len(), 30);
        assert_eq!(counters, FgoCounters::default());
        let ids: BTreeSet<u64> = out
            .iter()
            .flat_map(|f| f.associations.iter().filter_map(|a| a.track_id))
            .collect();
        assert_eq!(ids.len(), 2);
        let last = out.last().unwrap();
        let walker = last
            .tracks
            .iter()
            .find(|t| Some(t.track_id) == owner(last, 0))
            .unwrap();
        assert!(
            (walker.position[0] - 2.9).abs() < 0.05,
            "{:?}",
            walker.position
        );
        assert!(
            (walker.velocity[0] - 1.0).abs() < 0.1,
            "{:?}",
            walker.velocity
        );
        assert_eq!(walker.lifecycle, "confirmed");
    }

    #[test]
    fn undoes_a_swap_between_crossing_walkers() {
        // Walkers pass each other 0.4 m apart at frame 20; the engine swaps
        // its two tracks there.
        let input = frames(
            [[0.0, 0.0], [4.0, 0.4]],
            [[1.0, 0.0], [-1.0, 0.0]],
            40,
            |i, w| {
                if i < 21 {
                    w as u64 + 1
                } else {
                    2 - w as u64
                }
            },
        );
        let (out, counters) = run(input);
        assert_eq!(out.len(), 40);
        assert!(counters.swaps >= 1, "{counters:?}");
        for w in 0..2 {
            assert_eq!(
                owner(&out[0], w),
                owner(&out[39], w),
                "walker {w} changed identity"
            );
        }
        assert_ne!(owner(&out[0], 0), owner(&out[0], 1));
    }
}
