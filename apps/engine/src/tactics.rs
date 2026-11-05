//! Which kernel runs, and the evidence that decided it.
//!
//! The engine has one kernel per (regime, shape) pair and the choice is made
//! once, when the model loads: `Compute` then only asks the plan which path to
//! take. Two rules come from how the variants were measured:
//!
//! - a variant that lost on the clock is never selected again, even when it is
//!   numerically identical — the table says so and the test enforces it;
//! - a shape no kernel covers falls back explicitly, with the reason recorded,
//!   rather than silently taking a slow path.
//!
//! Every entry names the measurement that put it where it is, so a future
//! change has to argue with a number instead of with a comment.

use sha2::{Digest, Sha256};

use crate::model::Config;

/// How far a variant has got. `Rejected` is a permanent veto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maturity {
    /// Measured, correct, and in use.
    Accepted,
    /// Measured and correct, but behind the accepted variant.
    Verified,
    /// Correct or not, it lost: never selectable.
    Rejected,
}

/// The attention kernels the engine can launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionPath {
    /// Decode: positions spread over blocks, probabilities never leave registers.
    DecodeSplit,
    /// Prefill on the f16 tensor cores.
    PrefillMma,
    /// Prefill with keys and values staged, softmax kept online.
    PrefillTiled,
    /// One block per (head, query token), writing a full score grid.
    Serial,
    /// The host backend, which also writes a full grid.
    Host,
}

impl AttentionPath {
    /// Whether the path writes an `n_head × tokens × stride` score grid.
    pub fn needs_score_grid(self) -> bool {
        matches!(self, Self::Serial | Self::Host)
    }
}

/// Which half of a generation a variant belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    Decode,
    Prefill,
}

/// What a variant needs from the model to be legal. Shapes are exact, not
/// approximate: a kernel written for one tile is not launched for another.
#[derive(Debug, Clone, Copy)]
pub struct Guard {
    /// Exact head dimension the kernel is written for.
    pub head_dim: Option<usize>,
    /// `n_head` must be a multiple of this.
    pub heads_multiple: usize,
    /// Exact query-heads-per-kv-head, when the kernel assumes one.
    pub kv_group: Option<usize>,
}

impl Guard {
    /// Legal for any shape.
    pub const ANY: Guard = Guard {
        head_dim: None,
        heads_multiple: 1,
        kv_group: None,
    };

    /// The first reason this guard rejects `config`, if any.
    fn reject(&self, config: &Config) -> Option<&'static str> {
        if let Some(head_dim) = self.head_dim
            && config.head_dim != head_dim
        {
            return Some("head_dim mismatch");
        }
        if !config.n_head.is_multiple_of(self.heads_multiple) {
            return Some("head count is not a multiple the tile covers");
        }
        if let Some(group) = self.kv_group
            && config.n_head / config.n_head_kv != group
        {
            return Some("query-heads-per-kv-head mismatch");
        }
        None
    }
}

/// One kernel variant: what it needs, how far it got, and why.
pub struct Tactic {
    pub name: &'static str,
    pub regime: Regime,
    pub guard: Guard,
    /// `None` for a variant the table records but the engine does not have.
    pub path: Option<AttentionPath>,
    pub maturity: Maturity,
    /// Higher wins among variants whose guard matches.
    pub rank: u32,
    pub evidence: &'static str,
}

impl Tactic {
    fn selectable(&self) -> bool {
        self.path.is_some() && self.maturity != Maturity::Rejected
    }
}

/// Every attention variant this engine knows about, measured or refused.
pub const ATTENTION: &[Tactic] = &[
    Tactic {
        name: "attention.decode_split",
        regime: Regime::Decode,
        guard: Guard::ANY,
        path: Some(AttentionPath::DecodeSplit),
        maturity: Maturity::Accepted,
        rank: 10,
        evidence: "flat decode curve, 15.5/15.8/16.0/16.6 ms at ctx 0/128/512/1024 \
                   (was 15.4 rising to 26.2); ctx 1024 38.1 -> 60.1 tok/s",
    },
    Tactic {
        name: "attention.prefill_mma_f16",
        regime: Regime::Prefill,
        guard: Guard {
            head_dim: Some(128),
            heads_multiple: 2,
            kv_group: Some(8),
        },
        path: Some(AttentionPath::PrefillMma),
        maturity: Maturity::Accepted,
        rank: 10,
        evidence: "pp512 4.099 -> 4.856 tok/s with attention 25.5 -> 10.5 ms; logits \
                   delta to llama.cpp 0.321 -> 0.274 because f16 is their precision",
    },
    Tactic {
        name: "attention.prefill_tiled_f32",
        regime: Regime::Prefill,
        guard: Guard {
            head_dim: Some(128),
            heads_multiple: 2,
            kv_group: None,
        },
        path: Some(AttentionPath::PrefillTiled),
        maturity: Maturity::Verified,
        rank: 5,
        evidence: "pp512 4.099 tok/s, attention 25.1 ms: correct, superseded by the f16 \
                   tensor-core tile for the same shapes",
    },
    Tactic {
        name: "attention.prefill_serial",
        regime: Regime::Prefill,
        guard: Guard::ANY,
        path: Some(AttentionPath::Serial),
        maturity: Maturity::Verified,
        rank: 1,
        evidence: "one block per (head, token) writing the full score grid: correct for \
                   any shape, and the fallback when no tile matches",
    },
    Tactic {
        name: "attention.prefill_heads8",
        regime: Regime::Prefill,
        guard: Guard {
            head_dim: Some(128),
            heads_multiple: 8,
            kv_group: Some(8),
        },
        path: None,
        maturity: Maturity::Rejected,
        rank: 0,
        evidence: "eight heads per block needs 112 KB shared against the 99 KB limit of \
                   sm_89, so it was never written",
    },
];

/// Grouping policy for the projections that read the same activations.
///
/// q, k and v always share a launch: at 128 tokens the three together fill a
/// wave where q alone leaves four SMs idle. The feed-forward pair already fills
/// ten waves at 512 tokens, and past that the wider tile set only costs — at
/// 1024 tokens the grouped launch measured 2.40 ms against 2 × 0.70 ms separate.
const GROUP_FFN_MAX_TOKENS: usize = 512;
/// Group size that always shares a launch regardless of token count.
const GROUP_ALWAYS_MEMBERS: usize = 3;
/// Below this many tokens the row-at-a-time matmul quantises inline, so a
/// separate quantise pass would be one launch for nothing.
const PREPARE_MIN_TOKENS: usize = 8;

/// One variant that was considered for a regime.
#[derive(Debug, Clone)]
pub struct Considered {
    pub name: &'static str,
    pub chosen: bool,
    pub reason: &'static str,
}

/// The kernel a regime resolved to.
#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub path: AttentionPath,
    pub tactic: &'static str,
    /// Set when the general path runs because no specialized variant matched,
    /// with the reason the highest-ranked candidate gave.
    pub general: Option<&'static str>,
}

impl Choice {
    /// Whether this is the path that covers any shape, rather than a kernel
    /// written for this one.
    pub fn is_general(&self) -> bool {
        matches!(self.path, AttentionPath::Serial | AttentionPath::Host)
    }
}

/// Everything decided once at load: which kernel each regime runs, and what
/// that implies for scratch.
#[derive(Debug, Clone)]
pub struct KernelPlan {
    pub decode: Choice,
    pub prefill: Choice,
    pub considered: Vec<Considered>,
    /// Scratch bytes are sized from the plan, not guessed at the call site.
    pub prepare_min_tokens: usize,
    group_always_members: usize,
    group_max_tokens: usize,
    device: Option<String>,
}

impl KernelPlan {
    /// Resolve both regimes for this model on this backend.
    pub fn for_model(config: &Config, device: Option<&str>) -> Self {
        let mut considered = Vec::new();
        let (decode, prefill) = match device {
            None => {
                let host = Choice {
                    path: AttentionPath::Host,
                    tactic: "backend.host",
                    general: Some("the host backend runs the whole pass"),
                };
                (host, host)
            }
            Some(_) => (
                choose(Regime::Decode, config, &mut considered),
                choose(Regime::Prefill, config, &mut considered),
            ),
        };
        Self {
            decode,
            prefill,
            considered,
            prepare_min_tokens: PREPARE_MIN_TOKENS,
            group_always_members: GROUP_ALWAYS_MEMBERS,
            group_max_tokens: GROUP_FFN_MAX_TOKENS,
            device: device.map(str::to_owned),
        }
    }

    /// The path for a call of `tokens` tokens.
    pub fn attention(&self, tokens: usize) -> AttentionPath {
        if tokens == 1 {
            self.decode.path
        } else {
            self.prefill.path
        }
    }

    /// Floats of score grid this call needs, zero for the paths that keep
    /// probabilities in registers.
    pub fn score_floats(&self, heads: usize, tokens: usize, stride: usize) -> usize {
        if self.attention(tokens).needs_score_grid() {
            heads * tokens * stride
        } else {
            0
        }
    }

    /// Whether this group of projections should share one launch.
    pub fn group_matmul(&self, members: usize, tokens: usize) -> bool {
        members == self.group_always_members || tokens <= self.group_max_tokens
    }

    /// Regimes running the general path, with the reason no specialized kernel
    /// took them. Empty when every regime found a kernel written for it.
    pub fn fallbacks(&self) -> Vec<(&'static str, &'static str)> {
        [("decode", self.decode), ("prefill", self.prefill)]
            .into_iter()
            .filter_map(|(regime, choice)| choice.general.map(|reason| (regime, reason)))
            .collect()
    }

    /// Identity of the selection: two runs with the same value ran the same
    /// kernels with the same policy. The device name is a machine detail and
    /// stays out.
    pub fn identity(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.decode.tactic.as_bytes());
        hasher.update(b"\n");
        hasher.update(self.prefill.tactic.as_bytes());
        hasher.update(b"\n");
        hasher.update(self.prepare_min_tokens.to_string().as_bytes());
        hasher.update(b"\n");
        hasher.update(self.group_always_members.to_string().as_bytes());
        hasher.update(b"\n");
        hasher.update(self.group_max_tokens.to_string().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// One line per candidate: what was chosen, and what was not and why.
    pub fn explain(&self) -> String {
        let mut text = String::new();
        for candidate in &self.considered {
            let mark = if candidate.chosen {
                "chosen  "
            } else {
                "rejected"
            };
            text.push_str(&format!(
                "{mark} {}: {}\n",
                candidate.name, candidate.reason
            ));
        }
        for (regime, reason) in self.fallbacks() {
            text.push_str(&format!("fallback {regime}: {reason}\n"));
        }
        text
    }

    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }
}

/// Pick the best variant whose guard matches, recording every candidate.
///
/// `attention.prefill_serial` has an `ANY` guard, so a regime always resolves;
/// what varies is whether the path is the general one.
fn choose(regime: Regime, config: &Config, considered: &mut Vec<Considered>) -> Choice {
    let mut best: Option<&Tactic> = None;
    let mut verdicts: Vec<(&'static str, Option<&'static str>)> = Vec::new();
    let mut specialized_reason: Option<&'static str> = None;
    for tactic in ATTENTION.iter().filter(|tactic| tactic.regime == regime) {
        let rejection = tactic
            .guard
            .reject(config)
            .or_else(|| (tactic.maturity == Maturity::Rejected).then_some("maturity is rejected"))
            .or_else(|| (!tactic.selectable()).then_some("not implemented"));
        if rejection.is_some()
            && tactic.path != Some(AttentionPath::Serial)
            && specialized_reason.is_none()
        {
            specialized_reason = rejection;
        }
        if rejection.is_none() && best.is_none_or(|current| tactic.rank > current.rank) {
            best = Some(tactic);
        }
        verdicts.push((tactic.name, rejection));
    }
    let chosen = best.map(|tactic| tactic.name);
    for (name, rejection) in verdicts {
        let is_chosen = chosen == Some(name);
        considered.push(Considered {
            name,
            chosen: is_chosen,
            reason: match (is_chosen, rejection) {
                (true, _) => "selected",
                (false, Some(reason)) => reason,
                (false, None) => "a higher-ranked variant matches",
            },
        });
    }
    let tactic = best.expect("the general path matches every shape");
    let path = tactic.path.expect("a selectable tactic carries a path");
    Choice {
        path,
        tactic: tactic.name,
        general: path
            .needs_score_grid()
            .then(|| specialized_reason.unwrap_or("no specialized kernel for this regime")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model the engine ships with: 16 query heads over 2 kv heads, 128 dims.
    fn minicpm() -> Config {
        Config {
            n_layers: 42,
            n_embd: 2048,
            n_ff: 6144,
            n_head: 16,
            n_head_kv: 2,
            head_dim: 128,
            n_rot: 128,
            rope_base: 5e6,
            eps: 1e-6,
            n_vocab: 130_560,
        }
    }

    #[test]
    fn the_shipped_model_takes_the_measured_kernels() {
        let plan = KernelPlan::for_model(&minicpm(), Some("RTX 4050"));
        assert_eq!(plan.decode.path, AttentionPath::DecodeSplit);
        assert_eq!(plan.prefill.path, AttentionPath::PrefillMma);
        assert!(plan.fallbacks().is_empty());
        assert_eq!(plan.score_floats(16, 512, 512), 0);
    }

    #[test]
    fn a_shape_no_tile_covers_falls_back_with_a_reason() {
        let config = Config {
            head_dim: 64,
            ..minicpm()
        };
        let plan = KernelPlan::for_model(&config, Some("RTX 4050"));
        assert_eq!(plan.prefill.path, AttentionPath::Serial);
        assert_eq!(plan.fallbacks().len(), 1);
        assert!(plan.score_floats(16, 8, 8) > 0);
    }

    #[test]
    fn the_cpu_backend_has_no_gpu_plan() {
        let plan = KernelPlan::for_model(&minicpm(), None);
        assert_eq!(plan.attention(1), AttentionPath::Host);
        assert_eq!(plan.attention(64), AttentionPath::Host);
    }

    /// Every shape that reaches `choose` must leave with a legal variant.
    #[test]
    fn a_rejected_variant_is_never_selected() {
        for head_dim in [64, 128, 256] {
            for n_head in [8, 16, 32] {
                for n_head_kv in [1, 2, 4, 8] {
                    if n_head % n_head_kv != 0 {
                        continue;
                    }
                    let config = Config {
                        head_dim,
                        n_head,
                        n_head_kv,
                        ..minicpm()
                    };
                    let plan = KernelPlan::for_model(&config, Some("device"));
                    for choice in [plan.decode, plan.prefill] {
                        let tactic = ATTENTION
                            .iter()
                            .find(|tactic| tactic.name == choice.tactic)
                            .expect("chosen tactic is in the table");
                        assert_ne!(tactic.maturity, Maturity::Rejected, "{}", choice.tactic);
                        assert_eq!(tactic.path, Some(choice.path));
                    }
                }
            }
        }
    }

    /// The ledger is only worth having if every entry carries its evidence.
    #[test]
    fn every_variant_states_its_evidence() {
        for tactic in ATTENTION {
            assert!(
                tactic.evidence.len() > 40,
                "{} has no evidence",
                tactic.name
            );
            if tactic.maturity == Maturity::Rejected {
                assert!(
                    tactic.path.is_none(),
                    "{} is rejected but selectable",
                    tactic.name
                );
            }
            assert!(!tactic.selectable() || tactic.rank > 0, "{}", tactic.name);
        }
    }

    #[test]
    fn the_identity_moves_with_the_selection() {
        let plan = KernelPlan::for_model(&minicpm(), Some("device"));
        assert_eq!(plan.identity(), plan.identity());
        assert_eq!(plan.identity().len(), 64);
        let other = KernelPlan::for_model(&minicpm(), None);
        assert_ne!(plan.identity(), other.identity());
    }

    #[test]
    fn grouping_stops_where_it_measured_worse() {
        let plan = KernelPlan::for_model(&minicpm(), Some("device"));
        assert!(plan.group_matmul(3, 4096), "q|k|v always share a launch");
        assert!(plan.group_matmul(2, 512));
        assert!(!plan.group_matmul(2, 1024), "2.40 ms against 2 x 0.70 ms");
    }
}
