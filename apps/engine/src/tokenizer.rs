//! Tokenizers, as the file declares them.
//!
//! A GGUF names its vocabulary family in `tokenizer.ggml.model`: `gpt2` for the
//! byte-level BPE family (GPT-2, Qwen, MiniCPM), `llama` or `spm` for
//! SentencePiece (Llama 2, Mistral, Gemma). Both are read here; anything else is
//! refused by name rather than guessed at.
//!
//! The BPE path keeps two behaviours that are load-bearing and copied from
//! llama.cpp rather than guessed:
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
const PRE_MINICPM5: [&str; 2] = [
    r"\p{N}{1,3}",
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^ \t\n\x0b\f\r\p{L}\p{N}]+[\r\n]*|[ \t\n\x0b\f\r]*[\r\n]+|[ \t\n\x0b\f\r]+(?![^ \t\n\x0b\f\r])|[ \t\n\x0b\f\r]+",
];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2`.
const PRE_QWEN2: [&str; 2] = [
    r"\p{N}{1,3}",
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^ \t\n\x0b\f\r\p{L}\p{N}]+[\r\n]*|[ \t\n\x0b\f\r]*[\r\n]+|[ \t\n\x0b\f\r]+(?![^ \t\n\x0b\f\r])|[ \t\n\x0b\f\r]+",
];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN35`: a letter run also takes the Unicode
/// combining marks that follow it, and digits are cut one at a time. The
/// reference implements this as a hand-written scanner; the expression in its
/// comment is what the scanner decides, and it is applied here as a regex the
/// way the other pre-tokenizers are.
const PRE_QWEN35: [&str; 1] = [
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^ \t\n\x0b\f\r\p{L}\p{M}\p{N}]+[\r\n]*|[ \t\n\x0b\f\r]*[\r\n]+|[ \t\n\x0b\f\r]+(?![^ \t\n\x0b\f\r])|[ \t\n\x0b\f\r]+",
];

/// llama.cpp `LLAMA_VOCAB_PRE_TYPE_GPT2`, also used by `llama-bpe` models.
const PRE_GPT2: [&str; 1] = [
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^ \t\n\x0b\f\r\p{L}\p{N}]+|[ \t\n\x0b\f\r]+(?![^ \t\n\x0b\f\r])|[ \t\n\x0b\f\r]+",
];

/// The SentencePiece space marker.
const SPACE_MARKER: char = '\u{2581}';

const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// A slice of input: either literal text to tokenize or a special token id.
enum Chunk<'a> {
    Text(&'a str),
    Special(u32),
}

/// How this vocabulary turns text into ids. Boxed because the merge table is
/// large and the tokenizer itself is long-lived.
enum Encoder {
    Bpe(Box<BpeEncoder>),
    SentencePiece(Box<SentencePieceEncoder>),
}

/// Byte-level BPE: a merge table over the 256 byte symbols.
struct BpeEncoder {
    /// `(left_id, right_id) -> (rank, merged_id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_tokens: [u32; 256],
    alphabet: ByteAlphabet,
    patterns: Vec<Regex>,
}

/// SentencePiece: pieces with scores, merged by score, byte fallback.
struct SentencePieceEncoder {
    scores: Vec<f32>,
    add_space_prefix: bool,
    /// `Some(byte)` for a `<0xNN>` piece.
    byte_pieces: Vec<Option<u8>>,
}

pub struct Tokenizer {
    tokens: Vec<String>,
    token_ids: HashMap<String, u32>,
    encoder: Encoder,
    /// Longest first, so the first match is the longest special token.
    specials: Vec<(String, u32)>,
    special_ids: HashSet<u32>,
    bos: Option<u32>,
    eos: Option<u32>,
    family: &'static str,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, EngineError> {
        let tokens: Vec<String> = gguf
            .meta_str_array("tokenizer.ggml.tokens")?
            .into_iter()
            .map(str::to_string)
            .collect();
        let token_types = gguf.meta_i32_array("tokenizer.ggml.token_type")?;
        let token_ids: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();

        let family = gguf.meta_str("tokenizer.ggml.model").unwrap_or("gpt2");
        let (encoder, family) = match family {
            "gpt2" => {
                let patterns =
                    pre_tokenizer(gguf.meta_str("tokenizer.ggml.pre").unwrap_or("minicpm5"))?;
                (Self::bpe(gguf, &token_ids, patterns)?, "gpt2")
            }
            "llama" | "spm" => (
                Self::sentencepiece(gguf, &tokens, &token_ids)?,
                "sentencepiece",
            ),
            other => {
                return Err(EngineError::Unsupported(format!(
                    "tokenizer.ggml.model = {other:?}; this engine reads the gpt2 and \
                     sentencepiece families"
                )));
            }
        };

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

        Ok(Self {
            tokens,
            token_ids,
            encoder,
            specials,
            special_ids,
            bos: special_id(gguf, "bos_token_id"),
            eos: special_id(gguf, "eos_token_id"),
            family,
        })
    }

    fn bpe(
        gguf: &GgufFile,
        token_ids: &HashMap<String, u32>,
        patterns: Vec<Regex>,
    ) -> Result<Encoder, EngineError> {
        let merge_strings = gguf.meta_str_array("tokenizer.ggml.merges")?;
        let mut merges = HashMap::with_capacity(merge_strings.len());
        for (rank, entry) in merge_strings.iter().enumerate() {
            let (left, right) = entry.split_once(' ').ok_or_else(|| {
                EngineError::Format(format!("merge entry without separator: {entry:?}"))
            })?;
            let merged = format!("{left}{right}");
            let left_id = lookup(token_ids, left)?;
            let right_id = lookup(token_ids, right)?;
            let merged_id = lookup(token_ids, &merged)?;
            merges.insert((left_id, right_id), (rank as u32, merged_id));
        }

        let alphabet = ByteAlphabet::new();
        let mut byte_tokens = [0u32; 256];
        for (byte, slot) in byte_tokens.iter_mut().enumerate() {
            let symbol = alphabet.char_for(byte as u8).to_string();
            *slot = lookup(token_ids, &symbol)?;
        }

        Ok(Encoder::Bpe(Box::new(BpeEncoder {
            merges,
            byte_tokens,
            alphabet,
            patterns,
        })))
    }

    fn sentencepiece(
        gguf: &GgufFile,
        tokens: &[String],
        token_ids: &HashMap<String, u32>,
    ) -> Result<Encoder, EngineError> {
        let scores = gguf.meta_f32_array("tokenizer.ggml.scores")?;
        if scores.len() != tokens.len() {
            return Err(EngineError::Format(format!(
                "tokenizer.ggml.scores has {} entries for {} tokens",
                scores.len(),
                tokens.len()
            )));
        }
        let byte_pieces = tokens
            .iter()
            .map(|token| {
                let bytes = token.as_bytes();
                if bytes.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    u8::from_str_radix(&token[3..5], 16).ok()
                } else {
                    None
                }
            })
            .collect();
        let _ = token_ids;
        Ok(Encoder::SentencePiece(Box::new(SentencePieceEncoder {
            scores,
            add_space_prefix: gguf.meta_u64_opt("tokenizer.ggml.add_space_prefix") != Some(0),
            byte_pieces,
        })))
    }

    /// The piece a token id spells, for callers that need the marker a file
    /// declares (its BOS string, say) rather than a hardcoded one.
    pub fn token_of(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.token_ids.get(token).copied()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// The vocabulary family this file declared.
    pub fn family(&self) -> &'static str {
        self.family
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk in self.split_specials(text) {
            match chunk {
                Chunk::Special(id) => ids.push(id),
                Chunk::Text(slice) => match &self.encoder {
                    Encoder::Bpe(encoder) => {
                        for segment in pre_tokenize(&encoder.patterns, slice) {
                            self.push_bpe_segment(segment, &mut ids);
                        }
                    }
                    Encoder::SentencePiece(_) => self.push_sentencepiece(slice, &mut ids),
                },
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
        match &self.encoder {
            Encoder::Bpe(encoder) => {
                let mut bytes = Vec::with_capacity(token.len());
                for character in token.chars() {
                    match encoder.alphabet.byte_for(character) {
                        Some(byte) => bytes.push(byte),
                        None => bytes.extend_from_slice(character.to_string().as_bytes()),
                    }
                }
                bytes
            }
            Encoder::SentencePiece(encoder) => match encoder.byte_pieces[id as usize] {
                Some(byte) => vec![byte],
                None => token.replace(SPACE_MARKER, " ").into_bytes(),
            },
        }
    }

    /// Reconstruct the text a sequence spells.
    ///
    /// A SentencePiece encoder prepends the space marker, so the first piece
    /// carries a leading space that was never in the input; it is dropped here.
    /// Per-token [`Self::token_bytes`] keeps it, because a stream has to emit the
    /// bytes the piece actually stands for.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            bytes.extend_from_slice(&self.token_bytes(*id));
        }
        let strip_marker = matches!(&self.encoder, Encoder::SentencePiece(e) if e.add_space_prefix);
        if strip_marker && bytes.first() == Some(&b' ') {
            bytes.remove(0);
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

    fn push_bpe_segment(&self, segment: &str, ids: &mut Vec<u32>) {
        let Encoder::Bpe(encoder) = &self.encoder else {
            return;
        };
        let mut symbols: Vec<u32> = segment
            .bytes()
            .map(|byte| encoder.byte_tokens[byte as usize])
            .collect();
        while let Some((position, merged)) = best_merge(&encoder.merges, &symbols) {
            symbols.splice(position..position + 2, [merged]);
        }
        ids.extend(symbols);
    }

    /// SentencePiece encoding: characters first, then repeatedly merge the pair
    /// whose merged piece scores highest, exactly as llama.cpp's spm path does.
    /// Whatever cannot be merged falls back to one token per byte.
    fn push_sentencepiece(&self, text: &str, ids: &mut Vec<u32>) {
        let Encoder::SentencePiece(encoder) = &self.encoder else {
            return;
        };
        let scores = &encoder.scores;
        let add_space_prefix = encoder.add_space_prefix;
        let mut prepared = String::new();
        if add_space_prefix && !text.starts_with(' ') {
            prepared.push(SPACE_MARKER);
        }
        prepared.push_str(&text.replace(' ', &SPACE_MARKER.to_string()));

        let mut symbols: Vec<String> = prepared.chars().map(String::from).collect();
        loop {
            let mut best: Option<(usize, f32)> = None;
            for position in 0..symbols.len().saturating_sub(1) {
                let merged = format!("{}{}", symbols[position], symbols[position + 1]);
                let Some(&id) = self.token_ids.get(&merged) else {
                    continue;
                };
                let score = scores.get(id as usize).copied().unwrap_or(f32::MIN);
                if best.is_none_or(|(_, best_score)| score > best_score) {
                    best = Some((position, score));
                }
            }
            let Some((position, _)) = best else { break };
            let merged = format!("{}{}", symbols[position], symbols[position + 1]);
            symbols.splice(position..position + 2, [merged]);
        }

        for symbol in symbols {
            if let Some(&id) = self.token_ids.get(&symbol) {
                ids.push(id);
                continue;
            }
            for byte in symbol.bytes() {
                let piece = format!("<0x{byte:02X}>");
                if let Some(&id) = self.token_ids.get(&piece) {
                    ids.push(id);
                } else if let Some(unknown) = self.token_ids.get("<unk>") {
                    ids.push(*unknown);
                }
            }
        }
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

fn special_id(gguf: &GgufFile, name: &str) -> Option<u32> {
    gguf.meta_u64_opt(&format!("tokenizer.ggml.{name}"))
        .and_then(|value| u32::try_from(value).ok())
}

/// The pre-tokenizer a file asks for, by its llama.cpp name.
fn pre_tokenizer(name: &str) -> Result<Vec<Regex>, EngineError> {
    let patterns: &[&str] = match name {
        "minicpm5" => &PRE_MINICPM5,
        "qwen2" => &PRE_QWEN2,
        "qwen35" => &PRE_QWEN35,
        "gpt2" | "llama-bpe" | "default" | "" => &PRE_GPT2,
        other => {
            return Err(EngineError::Unsupported(format!(
                "tokenizer.ggml.pre = {other:?}; this engine implements the minicpm5, \
                 qwen2, qwen35, gpt2 and llama-bpe pre-tokenizers"
            )));
        }
    };
    patterns
        .iter()
        .map(|pattern| {
            Regex::new(pattern)
                .map_err(|error| EngineError::Format(format!("pre-tokenizer regex: {error}")))
        })
        .collect()
}

fn lookup(token_ids: &HashMap<String, u32>, token: &str) -> Result<u32, EngineError> {
    token_ids
        .get(token)
        .copied()
        .ok_or_else(|| EngineError::Format(format!("vocabulary is missing token {token:?}")))
}

/// The lowest-rank merge available in `symbols`.
fn best_merge(merges: &HashMap<(u32, u32), (u32, u32)>, symbols: &[u32]) -> Option<(usize, u32)> {
    let mut best: Option<(usize, u32, u32)> = None;
    for position in 0..symbols.len().saturating_sub(1) {
        let pair = (symbols[position], symbols[position + 1]);
        let Some((rank, merged)) = merges.get(&pair) else {
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

/// Each pattern splits the segments the previous pattern produced.
fn pre_tokenize<'a>(patterns: &[Regex], text: &'a str) -> Vec<&'a str> {
    let mut segments = vec![text];
    for pattern in patterns {
        let mut next = Vec::new();
        for segment in segments {
            split_keep_matches(pattern, segment, &mut next);
        }
        segments = next;
    }
    segments.into_iter().filter(|s| !s.is_empty()).collect()
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
