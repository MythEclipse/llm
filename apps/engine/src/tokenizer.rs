//! Byte-level BPE tokenizer matching llama.cpp's `minicpm5` pre-tokenizer.
//!
//! Two behaviours here are load-bearing and are copied from llama.cpp rather
//! than guessed:
//!
//! * the pre-tokenizer is **two regex passes applied in order** (`\p{N}{1,3}`
//!   first, so digit runs are already cut into ≤3-digit pieces), never one
//!   merged alternation;
//! * `\s` means the ASCII whitespace set, because llama.cpp uses `std::regex`
//!   ECMAScript semantics. Rust's Unicode `\s` would tokenize U+3000 differently.

use std::collections::{HashMap, HashSet};

use fancy_regex::Regex;

use crate::byte_level::ByteAlphabet;
use crate::error::EngineError;
use crate::gguf::GgufFile;

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_MINICPM5`, with `\s` spelled out as ASCII.
const PRE_TOKENIZER: [&str; 2] = [
    r"\p{N}{1,3}",
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^ \t\n\x0b\f\r\p{L}\p{N}]+[\r\n]*|[ \t\n\x0b\f\r]*[\r\n]+|[ \t\n\x0b\f\r]+(?![^ \t\n\x0b\f\r])|[ \t\n\x0b\f\r]+",
];

const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// A slice of input: either literal text to tokenize or a special token id.
enum Chunk<'a> {
    Text(&'a str),
    Special(u32),
}

pub struct Tokenizer {
    tokens: Vec<String>,
    token_ids: HashMap<String, u32>,
    /// `(left_id, right_id) -> (rank, merged_id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_tokens: [u32; 256],
    alphabet: ByteAlphabet,
    /// Longest first, so the first match is the longest special token.
    specials: Vec<(String, u32)>,
    special_ids: HashSet<u32>,
    patterns: Vec<Regex>,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, EngineError> {
        let tokens: Vec<String> = gguf
            .meta_str_array("tokenizer.ggml.tokens")?
            .into_iter()
            .map(str::to_string)
            .collect();
        let token_types = gguf.meta_i32_array("tokenizer.ggml.token_type")?;
        let merge_strings = gguf.meta_str_array("tokenizer.ggml.merges")?;

        let token_ids: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();

        let mut merges = HashMap::with_capacity(merge_strings.len());
        for (rank, entry) in merge_strings.iter().enumerate() {
            let (left, right) = entry.split_once(' ').ok_or_else(|| {
                EngineError::Format(format!("merge entry without separator: {entry:?}"))
            })?;
            let merged = format!("{left}{right}");
            let left_id = lookup(&token_ids, left)?;
            let right_id = lookup(&token_ids, right)?;
            let merged_id = lookup(&token_ids, &merged)?;
            merges.insert((left_id, right_id), (rank as u32, merged_id));
        }

        let alphabet = ByteAlphabet::new();
        let mut byte_tokens = [0u32; 256];
        for (byte, slot) in byte_tokens.iter_mut().enumerate() {
            let symbol = alphabet.char_for(byte as u8).to_string();
            *slot = lookup(&token_ids, &symbol)?;
        }

        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(id, _)| {
                matches!(
                    token_types.get(*id).copied(),
                    Some(TOKEN_TYPE_CONTROL) | Some(TOKEN_TYPE_USER_DEFINED)
                )
            })
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        specials.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));
        let special_ids: HashSet<u32> = specials.iter().map(|(_, id)| *id).collect();

        let patterns = PRE_TOKENIZER
            .iter()
            .map(|pattern| {
                Regex::new(pattern)
                    .map_err(|error| EngineError::Format(format!("pre-tokenizer regex: {error}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            tokens,
            token_ids,
            merges,
            byte_tokens,
            alphabet,
            specials,
            special_ids,
            patterns,
        })
    }

    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.token_ids.get(token).copied()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk in self.split_specials(text) {
            match chunk {
                Chunk::Special(id) => ids.push(id),
                Chunk::Text(slice) => {
                    for segment in self.pre_tokenize(slice) {
                        self.push_segment(segment, &mut ids);
                    }
                }
            }
        }
        ids
    }

    /// Raw bytes a token contributes when decoded. Streaming needs this to hold
    /// back a partial UTF-8 sequence instead of emitting a replacement character.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let token = match self.tokens.get(id as usize) {
            Some(token) => token,
            None => return Vec::new(),
        };
        if self.is_special(id) {
            return token.as_bytes().to_vec();
        }
        let mut bytes = Vec::with_capacity(token.len());
        for character in token.chars() {
            match self.alphabet.byte_for(character) {
                Some(byte) => bytes.push(byte),
                None => bytes.extend_from_slice(character.to_string().as_bytes()),
            }
        }
        bytes
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            bytes.extend_from_slice(&self.token_bytes(*id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    fn split_specials<'a>(&self, text: &'a str) -> Vec<Chunk<'a>> {
        let mut chunks = Vec::new();
        let mut pending = 0usize;
        let mut scan = 0usize;
        while scan < text.len() {
            let Some(offset) = text[scan..].find('<') else {
                break;
            };
            let at = scan + offset;
            match self.special_at(&text[at..]) {
                Some((length, id)) => {
                    if at > pending {
                        chunks.push(Chunk::Text(&text[pending..at]));
                    }
                    chunks.push(Chunk::Special(id));
                    pending = at + length;
                    scan = pending;
                }
                None => scan = at + 1,
            }
        }
        if pending < text.len() {
            chunks.push(Chunk::Text(&text[pending..]));
        }
        chunks
    }

    fn special_at(&self, tail: &str) -> Option<(usize, u32)> {
        self.specials
            .iter()
            .find(|(text, _)| tail.starts_with(text.as_str()))
            .map(|(text, id)| (text.len(), *id))
    }

    /// Each pattern splits the segments the previous pattern produced.
    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut segments = vec![text];
        for pattern in &self.patterns {
            let mut next = Vec::new();
            for segment in segments {
                split_keep_matches(pattern, segment, &mut next);
            }
            segments = next;
        }
        segments.into_iter().filter(|s| !s.is_empty()).collect()
    }

    fn push_segment(&self, segment: &str, ids: &mut Vec<u32>) {
        let mut symbols: Vec<u32> = segment
            .bytes()
            .map(|byte| self.byte_tokens[byte as usize])
            .collect();
        while let Some((position, merged)) = self.best_merge(&symbols) {
            symbols.splice(position..position + 2, [merged]);
        }
        ids.extend(symbols);
    }

    fn best_merge(&self, symbols: &[u32]) -> Option<(usize, u32)> {
        let mut best: Option<(usize, u32, u32)> = None;
        for position in 0..symbols.len().saturating_sub(1) {
            let pair = (symbols[position], symbols[position + 1]);
            let Some((rank, merged)) = self.merges.get(&pair) else {
                continue;
            };
            let improves = match best {
                Some((_, best_rank, _)) => *rank < best_rank,
                None => true,
            };
            if improves {
                best = Some((position, *rank, *merged));
            }
        }
        best.map(|(position, _, merged)| (position, merged))
    }
}

/// Turns per-token byte strings into displayable text.
///
/// A token can end in the middle of a UTF-8 sequence, so bytes are held back
/// until they form a complete character instead of emitting U+FFFD mid-word.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let complete = text.to_string();
                self.pending.clear();
                complete
            }
            Err(error) => {
                let valid = error.valid_up_to();
                let complete = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                if error.error_len().is_some() {
                    let remainder = String::from_utf8_lossy(&self.pending[valid..]).into_owned();
                    self.pending.clear();
                    return format!("{complete}{remainder}");
                }
                self.pending.drain(..valid);
                complete
            }
        }
    }

    /// Flush whatever is left at the end of a stream (may be a partial char).
    pub fn finish(&mut self) -> String {
        let remainder = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        remainder
    }
}

fn lookup(token_ids: &HashMap<String, u32>, token: &str) -> Result<u32, EngineError> {
    token_ids
        .get(token)
        .copied()
        .ok_or_else(|| EngineError::Format(format!("vocabulary is missing token {token:?}")))
}

/// Emit the matches of `pattern`, with the gaps between them as their own
/// segments — the same shape llama.cpp's `unicode_regex_split_stl` produces.
fn split_keep_matches<'a>(pattern: &Regex, text: &'a str, out: &mut Vec<&'a str>) {
    let mut cursor = 0usize;
    for found in pattern.find_iter(text) {
        let Ok(found) = found else { continue };
        if found.start() > cursor {
            out.push(&text[cursor..found.start()]);
        }
        if !found.as_str().is_empty() {
            out.push(found.as_str());
        }
        cursor = found.end();
    }
    if cursor < text.len() {
        out.push(&text[cursor..]);
    }
}
