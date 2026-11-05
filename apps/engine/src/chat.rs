//! ChatML rendering for MiniCPM5.
//!
//! The GGUF carries a 9 kB jinja template. Rather than embed a template engine,
//! the two shapes a chat needs are fixed here and pinned by tests: system/user
//! turns are plain, assistant turns carry an explicit (possibly empty) think
//! block, and the generation prompt closes with the assistant header.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

pub struct Turn<'a> {
    pub role: Role,
    pub content: &'a str,
}

pub const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

/// The template's `bos_token`. The tokenizer maps it to one id, so it costs a
/// single token and the model sees the sequence it was trained on.
pub const BOS: &str = "<s>";

/// Render the conversation up to (and including) the assistant header.
///
/// `thinking = false` emits the empty think block the template uses when
/// thinking is disabled, so replies arrive as direct answers.
///
/// The prompt starts with the beginning-of-sequence marker because the model's
/// own template does (`{{- bos_token }}`); leaving it out was a silent
/// divergence from every other front end. Measured on a four turn conversation
/// with the same questions: the recall answer changed from a Javanese rambling
/// sentence to "Kode proyek NX-42.", and the code answer from `print("Hello")` to
/// `print("Hello World!")`.
pub fn render(turns: &[Turn<'_>], thinking: bool) -> String {
    let mut prompt = String::from(BOS);
    for turn in turns {
        prompt.push_str("<|im_start|>");
        prompt.push_str(turn.role.as_str());
        prompt.push('\n');
        if turn.role == Role::Assistant && !turn.content.contains("</think>") {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        prompt.push_str(turn.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    if thinking {
        prompt.push_str("<think>\n");
    } else {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
    prompt
}
