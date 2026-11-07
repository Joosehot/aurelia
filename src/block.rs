//! The block: a 3D lattice of points, a ring accelerator on every four
//! neighbouring points of a layer, particles orbiting the rings.
//!
//! A ring on the square (i, j)-(i+1, j+1) of layer z is the circle through its
//! four corners, so its particles pass the corners at 45, 135, 225 and 315
//! degrees. A corner point is shared by up to four rings: that is where rings
//! touch and particles collide. A collision is clockwork: both particles
//! bounce back (their direction flips), some lift passes from the higher to
//! the lower, and the two rings' spins are pushed toward meshing
//! (counter-rotation), with omega_a - omega_b kept. Nothing is ever destroyed
//! or fused: the particle array never changes length.

use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Cfg {
    /// lattice points
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub particles_per_ring: usize,
    /// ring directions come from this seed; each reply has its own seed on top
    pub layout_seed: u64,
    pub spin_base: f32,
    pub spin_jitter: f32,
    pub spin_drag: f32,
    pub spin_max: f32,
    pub flip_share: f32,
    pub axle: f32,
    pub wave_speed: f32,
    pub wave_width: f32,
    pub wave_kick: f32,
    pub wave_fronts: usize,
    pub front_gap: usize,
    pub idle_steps: usize,
    pub rise: f32,
    pub against_rise: f32,
    pub limit_gain: f32,
    pub window: f32,
    pub lift_share: f32,
    pub mesh: f32,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            nx: 48,
            ny: 48,
            nz: 24,
            particles_per_ring: 3,
            layout_seed: 7,
            spin_base: 0.06,
            spin_jitter: 0.3,
            spin_drag: 0.04,
            spin_max: 0.9,
            flip_share: 0.15,
            axle: 0.02,
            wave_speed: 0.7,
            wave_width: 2.5,
            wave_kick: 0.05,
            wave_fronts: 12,
            front_gap: 24,
            idle_steps: 120,
            rise: 0.6,
            against_rise: 0.5,
            limit_gain: 4.0,
            window: 0.15,
            lift_share: 0.08,
            mesh: 0.3,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Particle {
    pub theta: f32,
    /// +1 runs with its ring, -1 against (after a bounce)
    pub dir: f32,
    /// how far it has risen toward its ring's limit
    pub h: f32,
    /// the fact branded onto it
    pub fact: Option<u32>,
    pub charge: f32,
    #[serde(skip)]
    pub hit_at: u32,
}

/// A small deterministic generator (splitmix64), so every reply replays from its seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct Block {
    pub cfg: Cfg,
    pub rx: usize,
    pub ry: usize,
    pub rz: usize,
    pub omega: Vec<f32>,
    pub base: Vec<f32>,
    pub limit: Vec<f32>,
    pub peak: Vec<i32>,
    pub parts: Vec<Particle>,
    pub dist: Vec<f32>,
    pub origin: (f32, f32, f32),
    pub collisions: u64,
    pub t: u32,
    sign: Vec<f32>,
}

/// The corner angle of corner k (0: (i+1, j+1), 1: (i, j+1), 2: (i, j), 3: (i+1, j)).
fn corner_angle(k: usize) -> f32 {
    FRAC_PI_4 + k as f32 * FRAC_PI_2
}

fn wrap(a: f32) -> f32 {
    let mut a = a % TAU;
    if a < 0.0 {
        a += TAU;
    }
    a
}

fn ang_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).abs() % TAU;
    d.min(TAU - d)
}

impl Block {
    pub fn new(cfg: Cfg) -> Block {
        let (rx, ry, rz) = (cfg.nx - 1, cfg.ny - 1, cfg.nz);
        let n = rx * ry * rz;
        let mut rng = Rng::new(cfg.layout_seed);
        let mut sign = vec![0.0; n];
        for z in 0..rz {
            for j in 0..ry {
                for i in 0..rx {
                    let checker = if (i + j + z) % 2 == 0 { 1.0 } else { -1.0 };
                    let flip = if rng.unit() < cfg.flip_share { -1.0 } else { 1.0 };
                    sign[(z * ry + j) * rx + i] = checker * flip;
                }
            }
        }
        let p = cfg.particles_per_ring;
        Block {
            rx,
            ry,
            rz,
            omega: vec![0.0; n],
            base: vec![cfg.spin_base; n],
            limit: vec![1.0; n],
            peak: vec![-1; n],
            parts: vec![Particle { dir: 1.0, ..Default::default() }; n * p],
            dist: vec![0.0; n],
            origin: (0.0, 0.0, 0.0),
            collisions: 0,
            t: 0,
            sign,
            cfg,
        }
    }

    pub fn rings(&self) -> usize {
        self.rx * self.ry * self.rz
    }

    pub fn ring(&self, i: usize, j: usize, z: usize) -> usize {
        (z * self.ry + j) * self.rx + i
    }

    pub fn coords(&self, r: usize) -> (usize, usize, usize) {
        (r % self.rx, (r / self.rx) % self.ry, r / (self.rx * self.ry))
    }

    /// The rings that share corner k of ring r, with the corner's index in each.
    #[cfg_attr(not(test), allow(dead_code))]
    fn sharing(&self, r: usize, k: usize) -> [(i64, usize); 3] {
        let (i, j, z) = self.coords(r);
        let (i, j) = (i as i64, j as i64);
        // the corner point (px, py)
        let (px, py) = match k {
            0 => (i + 1, j + 1),
            1 => (i, j + 1),
            2 => (i, j),
            _ => (i + 1, j),
        };
        // the four cells around the point and which corner of theirs it is
        let around = [((px - 1, py - 1), 0usize), ((px, py - 1), 1), ((px, py), 2), ((px - 1, py), 3)];
        let mut out = [(-1i64, 0usize); 3];
        let mut n = 0;
        for ((ci, cj), ck) in around {
            if (ci, cj) == (i, j) {
                continue;
            }
            if ci >= 0 && cj >= 0 && (ci as usize) < self.rx && (cj as usize) < self.ry {
                out[n] = (self.ring(ci as usize, cj as usize, z) as i64, ck);
            }
            n += 1;
        }
        out
    }

    /// Fresh motion for one reply: spins and angles from the seed, lifts at rest,
    /// then `idle_steps` of the block's own turning before the message's wave.
    pub fn stir(&mut self, seed: u64) {
        let mut rng = Rng::new(seed);
        let c = &self.cfg;
        for r in 0..self.rings() {
            let b = c.spin_base * (1.0 + c.spin_jitter * (2.0 * rng.unit() - 1.0));
            self.base[r] = b;
            self.omega[r] = self.sign[r] * b;
            self.peak[r] = -1;
        }
        for p in self.parts.iter_mut() {
            p.theta = TAU * rng.unit();
            p.dir = 1.0;
            p.h = 0.0;
            p.hit_at = 0;
        }
        self.collisions = 0;
        self.t = 0;
        for _ in 0..self.cfg.idle_steps {
            self.step(false);
        }
        for p in self.parts.iter_mut() {
            p.h = 0.0;
        }
        for r in 0..self.rings() {
            self.peak[r] = -1;
        }
        self.t = 0;
    }

    /// Ring limits from the particles' charges, and where the wave starts.
    pub fn charge(&mut self, origin: usize) {
        let p = self.cfg.particles_per_ring;
        for r in 0..self.rings() {
            let c = self.parts[r * p..(r + 1) * p].iter().map(|q| q.charge).fold(0.0f32, f32::max);
            self.limit[r] = 1.0 + self.cfg.limit_gain * c;
        }
        let (oi, oj, oz) = self.coords(origin);
        self.origin = (oi as f32 + 0.5, oj as f32 + 0.5, oz as f32);
        for r in 0..self.rings() {
            let (i, j, z) = self.coords(r);
            let (dx, dy, dz) = (i as f32 + 0.5 - self.origin.0, j as f32 + 0.5 - self.origin.1, z as f32 - self.origin.2);
            self.dist[r] = (dx * dx + dy * dy + dz * dz).sqrt();
        }
    }

    /// Steps until the last front has passed the far end of the block.
    pub fn wave_steps(&self) -> u32 {
        let far = self.dist.iter().cloned().fold(0.0f32, f32::max);
        (((self.cfg.wave_fronts - 1) * self.cfg.front_gap) as f32 + (far + self.cfg.wave_width) / self.cfg.wave_speed).ceil() as u32
    }

    /// One step of the whole block. Rings touch only inside their layer, so the layers
    /// turn in parallel (the same result on any number of threads); the axles between
    /// layers come after.
    pub fn step(&mut self, wave: bool) {
        let c = self.cfg.clone();
        self.t += 1;
        let t = self.t;
        let (rx, ry, pp) = (self.rx, self.ry, c.particles_per_ring);
        let layer = rx * ry;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(self.rz).max(1);
        let per = (self.rz + threads - 1) / threads * layer;
        let mut total = 0u64;
        std::thread::scope(|s| {
            let mut jobs = Vec::new();
            let chunks = self
                .omega
                .chunks_mut(per)
                .zip(self.parts.chunks_mut(per * pp))
                .zip(self.peak.chunks_mut(per))
                .zip(self.base.chunks(per))
                .zip(self.limit.chunks(per))
                .zip(self.dist.chunks(per))
                .zip(self.sign.chunks(per));
            for ((((((om, pa), pk), ba), li), di), sg) in chunks {
                let c = &c;
                jobs.push(s.spawn(move || {
                    let mut n = 0;
                    for l in 0..om.len() / layer {
                        let (r0, r1) = (l * layer, (l + 1) * layer);
                        let mut lay = Layer {
                            rx,
                            ry,
                            omega: &mut om[r0..r1],
                            parts: &mut pa[r0 * pp..r1 * pp],
                            peak: &mut pk[r0..r1],
                            base: &ba[r0..r1],
                            limit: &li[r0..r1],
                            dist: &di[r0..r1],
                            sign: &sg[r0..r1],
                        };
                        n += lay.step(c, t, wave);
                    }
                    n
                }));
            }
            for j in jobs {
                total += j.join().expect("a layer thread failed");
            }
        });
        // the axles: stacked rings share spin
        for r in layer..self.rings() {
            let (a, b) = (self.omega[r - layer].abs(), self.omega[r].abs());
            let d = c.axle * (a - b) * 0.5;
            self.omega[r - layer] = self.omega[r - layer].signum() * (a - d);
            self.omega[r] = self.omega[r].signum() * (b + d);
        }
        self.collisions += total;
    }

    /// Clockwork: both bounce, lift is shared, spins mesh. Destroys nothing.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn collide(&mut self, a: usize, b: usize, ra: usize, rb: usize) {
        let t = self.t;
        collide(&self.cfg, &mut self.parts, &mut self.omega, &self.limit, t, a, b, ra, rb);
        self.collisions += 1;
    }
    /// The top particle of a ring: how far it has risen, as a share of the limit.
    pub fn fill(&self, r: usize) -> f32 {
        let pp = self.cfg.particles_per_ring;
        self.parts[r * pp..(r + 1) * pp].iter().map(|p| p.h).fold(0.0, f32::max) / self.limit[r]
    }
}

fn collide(c: &Cfg, parts: &mut [Particle], omega: &mut [f32], limit: &[f32], t: u32, a: usize, b: usize, ra: usize, rb: usize) {
    parts[a].dir = -parts[a].dir;
    parts[b].dir = -parts[b].dir;
    parts[a].hit_at = t;
    parts[b].hit_at = t;
    let d = c.lift_share * (parts[a].h - parts[b].h) * 0.5;
    parts[a].h = (parts[a].h - d).min(limit[ra]);
    parts[b].h = (parts[b].h + d).min(limit[rb]);
    let (wa, wb) = (omega[ra], omega[rb]);
    omega[ra] = wa + c.mesh * (-wb - wa) * 0.5;
    omega[rb] = wb + c.mesh * (-wa - wb) * 0.5;
}

/// One layer of rings, turned on its own thread.
struct Layer<'a> {
    rx: usize,
    ry: usize,
    omega: &'a mut [f32],
    parts: &'a mut [Particle],
    peak: &'a mut [i32],
    base: &'a [f32],
    limit: &'a [f32],
    dist: &'a [f32],
    sign: &'a [f32],
}

impl Layer<'_> {
    /// The rings of this layer that share corner k of ring r, with the corner's index in each.
    fn sharing(&self, r: usize, k: usize) -> [(i64, usize); 3] {
        let (i, j) = ((r % self.rx) as i64, (r / self.rx) as i64);
        let (px, py) = match k {
            0 => (i + 1, j + 1),
            1 => (i, j + 1),
            2 => (i, j),
            _ => (i + 1, j),
        };
        let around = [((px - 1, py - 1), 0usize), ((px, py - 1), 1), ((px, py), 2), ((px - 1, py), 3)];
        let mut out = [(-1i64, 0usize); 3];
        let mut n = 0;
        for ((ci, cj), ck) in around {
            if (ci, cj) == (i, j) {
                continue;
            }
            if ci >= 0 && cj >= 0 && (ci as usize) < self.rx && (cj as usize) < self.ry {
                out[n] = ((cj as usize * self.rx + ci as usize) as i64, ck);
            }
            n += 1;
        }
        out
    }

    fn step(&mut self, c: &Cfg, t: u32, wave: bool) -> u64 {
        let n = self.omega.len();
        let pp = c.particles_per_ring;
        let mut hits = 0;
        // 1. the wave: every front passing a ring spins it up
        if wave {
            let (tf, gap, half) = (t as f32, c.front_gap as f32, c.wave_width * 0.5);
            for r in 0..n {
                let d = self.dist[r];
                let lo = ((tf - (d + half) / c.wave_speed) / gap).ceil().max(0.0) as i64;
                let hi = (((tf - (d - half) / c.wave_speed) / gap).floor() as i64).min(c.wave_fronts as i64 - 1);
                for f in lo..=hi {
                    let start = f as f32 * gap;
                    if tf >= start && (d - (tf - start) * c.wave_speed).abs() < half {
                        let s = self.omega[r].signum();
                        self.omega[r] += s * c.wave_kick;
                    }
                }
            }
        }
        // 2. drag back toward each ring's own pace
        for r in 0..n {
            let s = if self.omega[r] == 0.0 { self.sign[r] } else { self.omega[r].signum() };
            let mag = self.omega[r].abs();
            self.omega[r] = s * (mag - c.spin_drag * (mag - self.base[r])).clamp(0.0, c.spin_max);
        }
        // 3. particles orbit; a particle passing a corner meets whatever is at that corner
        for r in 0..n {
            for q in 0..pp {
                let idx = r * pp + q;
                let old = self.parts[idx].theta;
                let delta = self.omega[r] * self.parts[idx].dir;
                self.parts[idx].theta = wrap(old + delta);
                if self.parts[idx].hit_at == t {
                    continue;
                }
                let Some(k) = (0..4).find(|&k| crossed(old, delta, corner_angle(k))) else { continue };
                for (other, ok) in self.sharing(r, k) {
                    if other < 0 {
                        continue;
                    }
                    let other = other as usize;
                    let target = corner_angle(ok);
                    let hit = (0..pp)
                        .map(|w| other * pp + w)
                        .find(|&j| self.parts[j].hit_at != t && ang_dist(self.parts[j].theta, target) < c.window);
                    if let Some(j) = hit {
                        collide(c, self.parts, self.omega, self.limit, t, idx, j, r, other);
                        hits += 1;
                        break;
                    }
                }
            }
        }
        // 4. spin above a ring's own pace lifts its particles toward the ring's limit
        for r in 0..n {
            let ex = self.omega[r].abs() - self.base[r];
            if ex <= 0.0 {
                continue;
            }
            for q in 0..pp {
                let p = &mut self.parts[r * pp + q];
                let gain = if p.dir > 0.0 { 1.0 } else { c.against_rise };
                p.h = (p.h + c.rise * ex * gain).min(self.limit[r]);
                if wave && p.h >= self.limit[r] - 1e-6 && self.peak[r] < 0 {
                    self.peak[r] = t as i32;
                }
            }
        }
        hits
    }
}

/// Did an orbit step from `old` by `delta` pass the angle `at`?
fn crossed(old: f32, delta: f32, at: f32) -> bool {
    if delta == 0.0 {
        return false;
    }
    let delta = delta.clamp(-PI, PI);
    let rel = wrap(at - old);
    if delta > 0.0 {
        rel > 0.0 && rel <= delta
    } else {
        let rel = TAU - rel;
        rel > 0.0 && rel <= -delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Cfg {
        Cfg { nx: 12, ny: 12, nz: 3, idle_steps: 20, ..Default::default() }
    }

    #[test]
    fn particles_are_never_destroyed() {
        let mut b = Block::new(small());
        let n = b.parts.len();
        for (i, p) in b.parts.iter_mut().enumerate() {
            if i % 5 == 0 {
                p.fact = Some(i as u32);
                p.charge = 0.5;
            }
        }
        let branded: Vec<Option<u32>> = b.parts.iter().map(|p| p.fact).collect();
        for seed in 1..=5 {
            b.stir(seed);
            b.charge(0);
            for _ in 0..b.wave_steps() {
                b.step(true);
            }
            assert!(b.collisions > 0, "rings never touched");
            assert_eq!(b.parts.len(), n);
            assert_eq!(branded, b.parts.iter().map(|p| p.fact).collect::<Vec<_>>());
        }
    }

    #[test]
    fn a_collision_is_clockwork() {
        let mut b = Block::new(small());
        b.limit.iter_mut().for_each(|l| *l = 10.0);
        b.omega[0] = 0.3;
        b.omega[1] = 0.1;
        b.parts[0].h = 2.0;
        b.parts[3].h = 1.0;
        b.parts[0].dir = 1.0;
        b.parts[3].dir = 1.0;
        let (h, diff) = (b.parts[0].h + b.parts[3].h, b.omega[0] - b.omega[1]);
        b.collide(0, 3, 0, 1);
        assert_eq!(b.parts[0].dir, -1.0);
        assert_eq!(b.parts[3].dir, -1.0);
        assert!((b.parts[0].h + b.parts[3].h - h).abs() < 1e-6, "lift is passed, not made or lost");
        assert!((b.omega[0] - b.omega[1] - diff).abs() < 1e-6, "spins mesh, omega_a - omega_b kept");
        assert!(b.parts[0].h < 2.0 && b.parts[3].h > 1.0);
    }

    #[test]
    fn rings_touch_at_shared_corners() {
        let b = Block::new(small());
        let r = b.ring(3, 3, 0);
        let s: Vec<usize> = b.sharing(r, 0).iter().filter(|x| x.0 >= 0).map(|x| x.0 as usize).collect();
        assert_eq!(s.len(), 3);
        assert!(s.contains(&b.ring(4, 3, 0)) && s.contains(&b.ring(3, 4, 0)) && s.contains(&b.ring(4, 4, 0)));
    }
}
