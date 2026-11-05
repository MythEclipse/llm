//! Where the forward pass keeps its scratch, planned instead of allocated.
//!
//! Every buffer the layer loop touches is declared once, with the span of the
//! pass it lives in. Two buffers whose spans do not overlap share one region, so
//! the plan is smaller than the sum of its parts and — more importantly — fixed:
//! one allocation per session, and nothing allocated while a token is being
//! generated.
//!
//! The span table is also the invariant. A buffer read at phase 3 and another
//! written at phase 3 may not share a region, and [`BufferPlan::plan`] refuses to
//! place them there; a lifetime declared wrong shows up as a wrong number in the
//! golden fixtures rather than as a mystery.

use sha2::{Digest, Sha256};

use crate::model::Config;

/// Floats per alignment step. Kernels load `float4`/`half2`, so a view has to
/// start on a sixteen-byte boundary.
const ALIGN: usize = 4;

/// Phases of one forward pass, in order. A buffer's span is the range of phases
/// between its first and last use.
const NORM_ATTN: u32 = 0;
const QKV: u32 = 1;
const ROPE: u32 = 2;
const ATTEND: u32 = 3;
const OUT: u32 = 4;
const NORM_FFN: u32 = 5;
const FFN_UP: u32 = 6;
const ACT: u32 = 7;
const DOWN: u32 = 8;
const LOGITS: u32 = 9;

/// One buffer the pass needs.
#[derive(Debug, Clone)]
pub struct BufferRequest {
    pub name: &'static str,
    /// Size in f32 values.
    pub floats: usize,
    pub first_use: u32,
    pub last_use: u32,
}

impl BufferRequest {
    /// Whether the two are live at the same time, and therefore may not share.
    fn overlaps(&self, other: &Self) -> bool {
        !(self.last_use < other.first_use || other.last_use < self.first_use)
    }
}

/// Offsets for a set of requests, and the high-water mark.
#[derive(Debug, Clone)]
pub struct BufferPlan {
    names: Vec<&'static str>,
    sizes: Vec<usize>,
    offsets: Vec<usize>,
    floats: usize,
}

impl BufferPlan {
    /// Place every request at the lowest aligned offset no live neighbour holds.
    pub fn plan(requests: Vec<BufferRequest>) -> Self {
        let mut order: Vec<usize> = (0..requests.len()).collect();
        order.sort_by_key(|&index| (requests[index].first_use, requests[index].name));
        let mut offsets = vec![0usize; requests.len()];
        let mut placed: Vec<usize> = Vec::with_capacity(requests.len());
        for &index in &order {
            let request = &requests[index];
            if request.floats == 0 {
                placed.push(index);
                continue;
            }
            let mut offset = 0;
            loop {
                offset = align_up(offset);
                let clash = placed.iter().find_map(|&other| {
                    let neighbour = &requests[other];
                    let neighbour_offset = offsets[other];
                    (neighbour.floats > 0
                        && neighbour.overlaps(request)
                        && offset < neighbour_offset + neighbour.floats
                        && neighbour_offset < offset + request.floats)
                        .then_some(neighbour_offset + neighbour.floats)
                });
                match clash {
                    Some(next) => offset = next,
                    None => break,
                }
            }
            offsets[index] = offset;
            placed.push(index);
        }
        let floats = requests
            .iter()
            .zip(&offsets)
            .map(|(request, offset)| offset + request.floats)
            .max()
            .unwrap_or(0);
        Self {
            names: requests.iter().map(|request| request.name).collect(),
            sizes: requests.iter().map(|request| request.floats).collect(),
            offsets,
            floats,
        }
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|planned| *planned == name)
    }

    pub fn offset(&self, index: usize) -> usize {
        self.offsets[index]
    }

    pub fn size(&self, index: usize) -> usize {
        self.sizes[index]
    }

    /// Floats the whole plan needs, which is the size of the one allocation.
    pub fn floats(&self) -> usize {
        self.floats
    }

    pub fn bytes(&self) -> usize {
        self.floats * std::mem::size_of::<f32>()
    }

    /// Floats saved by letting buffers share a region.
    pub fn reused_floats(&self) -> usize {
        self.sizes.iter().sum::<usize>() - self.floats
    }

    /// Identity of the layout: same value, same offsets, same bytes.
    pub fn identity(&self) -> String {
        let mut hasher = Sha256::new();
        for (index, name) in self.names.iter().enumerate() {
            hasher.update(name.as_bytes());
            hasher.update(b":");
            hasher.update(self.sizes[index].to_string().as_bytes());
            hasher.update(b"@");
            hasher.update(self.offsets[index].to_string().as_bytes());
            hasher.update(b"\n");
        }
        format!("{:x}", hasher.finalize())
    }
}

fn align_up(offset: usize) -> usize {
    offset.div_ceil(ALIGN) * ALIGN
}

/// The buffers one forward pass of `tokens` tokens needs, with their spans.
///
/// `score_grid` comes from the kernel plan: the tiled and split paths keep
/// probabilities in registers, so their grid is not allocated at all.
pub fn scratch_requests(
    config: &Config,
    tokens: usize,
    capacity: usize,
    score_grid: bool,
) -> Vec<BufferRequest> {
    let embd = tokens * config.n_embd;
    let kv = tokens * config.kv_dim();
    let ffn = tokens * config.n_ff;
    vec![
        // The hidden state, live from the embedding to the last residual.
        BufferRequest {
            name: "x",
            floats: embd,
            first_use: NORM_ATTN,
            last_use: LOGITS,
        },
        BufferRequest {
            name: "logits",
            floats: config.n_vocab,
            first_use: LOGITS,
            last_use: LOGITS,
        },
        // One row: the output norm only ever normalises the last token.
        BufferRequest {
            name: "output_norm",
            floats: config.n_embd,
            first_use: LOGITS,
            last_use: LOGITS,
        },
        BufferRequest {
            name: "normed_attn",
            floats: embd,
            first_use: NORM_ATTN,
            last_use: QKV,
        },
        BufferRequest {
            name: "q",
            floats: embd,
            first_use: QKV,
            last_use: ROPE,
        },
        BufferRequest {
            name: "k",
            floats: kv,
            first_use: QKV,
            last_use: ROPE,
        },
        BufferRequest {
            name: "v",
            floats: kv,
            first_use: QKV,
            last_use: ROPE,
        },
        BufferRequest {
            name: "attn",
            floats: embd,
            first_use: ATTEND,
            last_use: OUT,
        },
        BufferRequest {
            name: "scores",
            floats: if score_grid {
                config.n_head * tokens * capacity
            } else {
                0
            },
            first_use: ATTEND,
            last_use: ATTEND,
        },
        BufferRequest {
            name: "projected",
            floats: embd,
            first_use: OUT,
            last_use: DOWN,
        },
        BufferRequest {
            name: "normed_ffn",
            floats: embd,
            first_use: NORM_FFN,
            last_use: FFN_UP,
        },
        BufferRequest {
            name: "gate",
            floats: ffn,
            first_use: FFN_UP,
            last_use: ACT,
        },
        BufferRequest {
            name: "up",
            floats: ffn,
            first_use: FFN_UP,
            last_use: ACT,
        },
        BufferRequest {
            name: "activated",
            floats: ffn,
            first_use: ACT,
            last_use: DOWN,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(name: &'static str, floats: usize, first: u32, last: u32) -> BufferRequest {
        BufferRequest {
            name,
            floats,
            first_use: first,
            last_use: last,
        }
    }

    #[test]
    fn buffers_that_never_coincide_share_a_region() {
        let plan = BufferPlan::plan(vec![
            request("first", 100, 0, 1),
            request("second", 100, 2, 3),
        ]);
        assert_eq!(plan.floats(), 100, "one region serves both");
        assert_eq!(plan.reused_floats(), 100);
        assert_eq!(plan.offset(0), plan.offset(1));
    }

    #[test]
    fn buffers_that_coincide_do_not() {
        let plan = BufferPlan::plan(vec![
            request("first", 100, 0, 1),
            request("second", 100, 1, 2),
        ]);
        assert_eq!(plan.floats(), 200);
        assert_ne!(plan.offset(0), plan.offset(1));
    }

    #[test]
    fn a_shared_edge_is_still_an_overlap() {
        // Written at 3 and read at 3 is the same phase, not two.
        let plan = BufferPlan::plan(vec![
            request("writer", 64, 3, 3),
            request("reader", 64, 3, 3),
        ]);
        assert_eq!(plan.floats(), 128);
    }

    #[test]
    fn a_small_dead_buffer_is_reused_by_a_later_large_one() {
        let plan = BufferPlan::plan(vec![request("small", 16, 0, 0), request("large", 64, 1, 1)]);
        assert_eq!(plan.floats(), 64, "the large one takes the region over");
    }

    #[test]
    fn every_offset_is_aligned_for_vector_loads() {
        let plan = BufferPlan::plan(vec![
            request("a", 3, 0, 0),
            request("b", 5, 0, 0),
            request("c", 7, 0, 0),
        ]);
        for index in 0..3 {
            assert_eq!(plan.offset(index) % ALIGN, 0, "buffer {index}");
        }
    }

    #[test]
    fn no_placed_pair_overlaps_in_memory_while_live() {
        let requests = scratch_requests(&crate::model::tests::minicpm(), 64, 512, true);
        let plan = BufferPlan::plan(requests.clone());
        for (index, left) in requests.iter().enumerate() {
            for (other, right) in requests.iter().enumerate().skip(index + 1) {
                if !left.overlaps(right) || left.floats == 0 || right.floats == 0 {
                    continue;
                }
                let (left_start, left_end) = (plan.offset(index), plan.offset(index) + left.floats);
                let (right_start, right_end) =
                    (plan.offset(other), plan.offset(other) + right.floats);
                assert!(
                    left_end <= right_start || right_end <= left_start,
                    "{} and {} share memory while both are live",
                    left.name,
                    right.name
                );
            }
        }
    }

    #[test]
    fn the_shipped_model_saves_regions_and_plans_a_grid_free_layout() {
        let config = crate::model::tests::minicpm();
        let with_grid = BufferPlan::plan(scratch_requests(&config, 1024, 8192, true));
        let without = BufferPlan::plan(scratch_requests(&config, 1024, 8192, false));
        assert!(without.floats() < with_grid.floats());
        assert!(without.reused_floats() > 0);
        assert_eq!(without.identity(), without.identity());
        assert_ne!(with_grid.identity(), without.identity());
    }
}
