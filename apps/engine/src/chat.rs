//! Chat prompt rendering, for the template a file declares.
//!
//! A GGUF carries its chat template as a jinja string in
//! `tokenizer.chat_template`. Embedding a template engine to run it is not worth
//! it here, but ignoring it is worse: the same conversation sent as ChatML to a
//! model trained on Llama-3 headers produces a wrong prompt and a bad answer.
//!
//! So the template string is classified by the markers it contains, and the
//! shapes that cover most open models are rendered natively. A template this
//! engine does not recognise is refused by name instead of being silently
//! rendered as ChatML.

use crate::error::EngineError;
use crate::gguf::GgufFile;
use crate::tokenizer::Tokenizer;

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

/// The template's `bos_token`, used by [`ChatFormat::chatml`] when no file is
/// open. Real loads take it from the tokenizer's own metadata.
pub const BOS: &str = "<s>";

/// The conversation shapes this engine can render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// `<|im_start|>role ... <|im_end|>` — Qwen, MiniCPM, many others.
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|> ... <|eot_id|>` — Llama 3.
    Llama3,
    /// `[INST] ... [/INST]` — Mistral and its derivatives.
    Mistral,
}

impl Template {
    pub fn name(self) -> &'static str {
        match self {
            Self::ChatMl => "chatml",
            Self::Llama3 => "llama3",
            Self::Mistral => "mistral",
        }
    }

    /// Which shape a declared template string produces, decided by its markers.
    pub fn detect(declared: &str) -> Result<Self, EngineError> {
        if declared.contains("<|im_start|>") {
            Ok(Self::ChatMl)
        } else if declared.contains("<|start_header_id|>") {
            Ok(Self::Llama3)
        } else if declared.contains("[INST]") {
            Ok(Self::Mistral)
        } else if declared.is_empty() {
            Err(EngineError::MissingMetadata(
                "tokenizer.chat_template".into(),
            ))
        } else {
            Err(EngineError::Unsupported(format!(
                "chat template is neither chatml, llama3 nor mistral: {}",
                declared.chars().take(80).collect::<String>()
            )))
        }
    }
}

/// Everything needed to turn a conversation into a prompt: the shape and the
/// beginning-of-sequence marker the file asks for.
#[derive(Debug, Clone)]
pub struct ChatFormat {
    template: Template,
    bos: String,
    /// Whether the declared template wraps answers in a reasoning block. Only
    /// the models that were trained with one declare `<think>`; emitting it for
    /// every ChatML model would prepend two tokens the model never saw.
    thinking: bool,
}

impl ChatFormat {
    /// ChatML with `<s>`, which is what the tests and the shipped model use.
    pub fn chatml() -> Self {
        Self {
            template: Template::ChatMl,
            bos: BOS.to_string(),
            thinking: true,
        }
    }

    /// Read the template and the BOS marker out of a file.
    pub fn from_gguf(gguf: &GgufFile, tokenizer: &Tokenizer) -> Result<Self, EngineError> {
        let declared = gguf.meta_str("tokenizer.chat_template").unwrap_or("");
        let bos = tokenizer
            .bos_id()
            .and_then(|id| tokenizer.token_of(id))
            .unwrap_or(BOS)
            .to_string();
        Ok(Self {
            template: Template::detect(declared)?,
            bos,
            thinking: declared.contains("<think>"),
        })
    }

    pub fn template(&self) -> Template {
        self.template
    }

    pub fn bos(&self) -> &str {
        &self.bos
    }

    /// Render the conversation up to (and including) the assistant header.
    ///
    /// `thinking` only affects ChatML: it decides whether the closing think block
    /// is empty (a direct answer) or left open (the model reasons first). The
    /// other shapes have no such marker.
    pub fn render(&self, turns: &[Turn<'_>], thinking: bool) -> String {
        match self.template {
            Template::ChatMl => self.render_chatml(turns, thinking),
            Template::Llama3 => self.render_llama3(turns),
            Template::Mistral => self.render_mistral(turns),
        }
    }

    /// The prompt starts with the beginning-of-sequence marker because the
    /// model's own template does; leaving it out was a silent divergence from
    /// every other front end. Measured on a four turn conversation with the same
    /// questions: the recall answer changed from a Javanese rambling sentence to
    /// "Kode proyek NX-42.", and the code answer from `print("Hello")` to
    /// `print("Hello World!")`.
    fn render_chatml(&self, turns: &[Turn<'_>], thinking: bool) -> String {
        let mut prompt = self.bos.clone();
        for turn in turns {
            prompt.push_str("<|im_start|>");
            prompt.push_str(turn.role.as_str());
            prompt.push('\n');
            if self.thinking && turn.role == Role::Assistant && !turn.content.contains("</think>") {
                prompt.push_str("<think>\n\n</think>\n\n");
            }
            prompt.push_str(turn.content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        if self.thinking {
            if thinking {
                prompt.push_str("<think>\n");
            } else {
                prompt.push_str("<think>\n\n</think>\n\n");
            }
        }
        prompt
    }

    fn render_llama3(&self, turns: &[Turn<'_>]) -> String {
        let mut prompt = self.bos.clone();
        for turn in turns {
            prompt.push_str("<|start_header_id|>");
            prompt.push_str(turn.role.as_str());
            prompt.push_str("<|end_header_id|>\n\n");
            prompt.push_str(turn.content);
            prompt.push_str("<|eot_id|>");
        }
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        prompt
    }

    /// Mistral has no system role: the system turn is folded into the first user
    /// message, and every instruction block closes with `[/INST]`.
    fn render_mistral(&self, turns: &[Turn<'_>]) -> String {
        let mut prompt = self.bos.clone();
        let mut pending_system: Option<&str> = None;
        let mut open = false;
        for turn in turns {
            match turn.role {
                Role::System => pending_system = Some(turn.content),
                Role::User => {
                    prompt.push_str("[INST] ");
                    if let Some(system) = pending_system.take() {
                        prompt.push_str(system);
                        prompt.push_str("\n\n");
                    }
                    prompt.push_str(turn.content);
                    prompt.push_str(" [/INST]");
                    open = true;
                }
                Role::Assistant => {
                    prompt.push(' ');
                    prompt.push_str(turn.content);
                    open = false;
                }
            }
        }
        if !open {
            prompt.push_str("[INST] [/INST]");
        }
        prompt
    }
}

/// ChatML with the shipped model's marker; kept as the plain entry point.
pub fn render(turns: &[Turn<'_>], thinking: bool) -> String {
    ChatFormat::chatml().render(turns, thinking)
}
