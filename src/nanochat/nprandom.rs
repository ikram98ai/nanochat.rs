//! NumPy-compatible seeded permutation.
//!
//! The reference shuffles every task dataset with
//! `np.random.default_rng(seed).permutation(n)` (the same order as HuggingFace
//! `Dataset.shuffle(seed=seed)`). Reproducing that order exactly needs NumPy's
//! SeedSequence (seed -> PCG64 state), the PCG64 generator (XSL-RR 128/64) and
//! Generator.shuffle's masked rejection sampling.

const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

const PCG_MULT: u128 = 0x2360ED051FC65DA44385DF649FCCF645;

/// np.random.SeedSequence(seed).generate_state(4, np.uint64)
fn seed_sequence_state(seed: u64) -> [u64; 4] {
    // the entropy as little-endian 32-bit words (at least one)
    let mut entropy = vec![seed as u32];
    if seed >> 32 != 0 {
        entropy.push((seed >> 32) as u32);
    }

    let mut hash_const = INIT_A;
    let mut hashmix = |value: u32| -> u32 {
        let mut value = value ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        value = value.wrapping_mul(hash_const);
        value ^ (value >> XSHIFT)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let result = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> XSHIFT)
    };

    // mix the entropy into the pool
    let mut pool = [0u32; POOL_SIZE];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src]));
            }
        }
    }
    // (entropy longer than the pool would be mixed in here; a u64 seed never is)

    // generate 8 x 32-bit words, viewed as 4 little-endian 64-bit words
    let mut hash_const = INIT_B;
    let mut words = [0u32; 2 * 4];
    for (i, word) in words.iter_mut().enumerate() {
        let mut value = pool[i % POOL_SIZE] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        value = value.wrapping_mul(hash_const);
        *word = value ^ (value >> XSHIFT);
    }
    let mut state = [0u64; 4];
    for (i, s) in state.iter_mut().enumerate() {
        *s = words[2 * i] as u64 | (words[2 * i + 1] as u64) << 32;
    }
    state
}

/// np.random.PCG64, the bit generator behind np.random.default_rng.
pub struct Pcg64 {
    state: u128,
    inc: u128,
    /// second half of the last 64-bit draw, served by the next 32-bit draw
    buffered_u32: Option<u32>,
}

impl Pcg64 {
    /// np.random.PCG64(seed)
    pub fn new(seed: u64) -> Self {
        let s = seed_sequence_state(seed);
        let initstate = (s[0] as u128) << 64 | s[1] as u128;
        let initseq = (s[2] as u128) << 64 | s[3] as u128;
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            buffered_u32: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    pub fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.buffered_u32.take() {
            return value;
        }
        let next = self.next_u64();
        self.buffered_u32 = Some((next >> 32) as u32);
        next as u32
    }

    /// Uniform integer in [0, max] by masked rejection sampling (random_interval).
    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        // smallest bit mask >= max
        let mask = u64::MAX >> max.leading_zeros();
        loop {
            let value = if max <= u32::MAX as u64 {
                self.next_u32() as u64 & mask
            } else {
                self.next_u64() & mask
            };
            if value <= max {
                return value;
            }
        }
    }
}

/// np.random.default_rng(seed).permutation(n)
pub fn permutation(seed: u64, n: usize) -> Vec<usize> {
    let mut rng = Pcg64::new(seed);
    let mut arr: Vec<usize> = (0..n).collect();
    // Generator.shuffle: Fisher-Yates from the back
    for i in (1..n).rev() {
        let j = rng.interval(i as u64) as usize;
        arr.swap(i, j);
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    // goldens from numpy 2.5.4

    #[test]
    fn test_seed_sequence_and_raw_stream() {
        assert_eq!(
            seed_sequence_state(42),
            [
                0x9f1e2e6dcd540ab7,
                0xd57873dc79fb94b6,
                0x7d282a1b64d420b7,
                0x336579714692d5ff
            ]
        );
        let mut rng = Pcg64::new(42);
        let raw: Vec<u64> = (0..3).map(|_| rng.next_u64()).collect();
        assert_eq!(
            raw,
            [0xc621fbcd16d92688, 0x705a5661a791ffc1, 0xdbcd12c26eda1624]
        );
    }

    #[test]
    fn test_small_permutations() {
        assert_eq!(permutation(42, 10), [5, 6, 0, 7, 3, 2, 4, 9, 1, 8]);
        assert_eq!(permutation(42, 0), Vec::<usize>::new());
        assert_eq!(permutation(42, 1), [0]);
        assert_eq!(permutation(42, 2), [1, 0]);
        assert_eq!(permutation(0, 7), [2, 4, 3, 6, 5, 0, 1]);
        assert_eq!(
            permutation(1337, 20),
            [12, 19, 11, 6, 7, 9, 3, 10, 13, 5, 18, 14, 16, 2, 8, 4, 15, 17, 0, 1]
        );
    }

    /// sum((i + 1) * p[i]) over the permutation, wrapping in u64
    fn checksum(p: &[usize]) -> u64 {
        p.iter().enumerate().fold(0u64, |acc, (i, &v)| {
            acc.wrapping_add((i as u64 + 1).wrapping_mul(v as u64))
        })
    }

    #[test]
    fn test_dataset_sized_permutations() {
        // (seed, n, first 6, last 3, checksum)
        let cases: [(u64, usize, [usize; 6], [usize; 3], u64); 4] = [
            (
                42,
                2376,
                [658, 1619, 1811, 47, 1494, 1273],
                [1572, 1633, 1672],
                CHECKSUMS[0],
            ),
            (
                42,
                100000,
                [45262, 49850, 30394, 90177, 95569, 10900],
                [5668, 22113, 75400],
                CHECKSUMS[1],
            ),
            (
                42,
                460341,
                [21697, 61600, 155055, 325090, 320249, 306258],
                [131009, 129997, 75400],
                CHECKSUMS[2],
            ),
            (
                7,
                70000,
                [15808, 48179, 11705, 6440, 12002, 28192],
                [63842, 27285, 16809],
                CHECKSUMS[3],
            ),
        ];
        for (seed, n, head, tail, sum) in cases {
            let p = permutation(seed, n);
            assert_eq!(p[..6], head, "head of permutation({seed}, {n})");
            assert_eq!(p[n - 3..], tail, "tail of permutation({seed}, {n})");
            assert_eq!(checksum(&p), sum, "checksum of permutation({seed}, {n})");
        }
    }

    const CHECKSUMS: [u64; 4] = [
        3335535977,
        249157033702609,
        24370630385377046,
        85727311659269,
    ];
}
