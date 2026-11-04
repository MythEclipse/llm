//! ChatML rendering pinned to the shapes the GGUF template produces.

use llm_engine::chat::{DEFAULT_SYSTEM, Role, Turn, render};

#[test]
fn single_user_turn_closes_with_assistant_header() {
    let prompt = render(
        &[
            Turn {
                role: Role::System,
                content: DEFAULT_SYSTEM,
            },
            Turn {
                role: Role::User,
                content: "Halo",
            },
        ],
        false,
    );

    assert_eq!(
        prompt,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\nHalo<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn assistant_history_without_think_block_gets_an_empty_one() {
    let prompt = render(
        &[
            Turn {
                role: Role::User,
                content: "2+2?",
            },
            Turn {
                role: Role::Assistant,
                content: "4",
            },
            Turn {
                role: Role::User,
                content: "dan 3+3?",
            },
        ],
        false,
    );

    assert!(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>\n"));
}

#[test]
fn assistant_history_with_think_block_is_left_alone() {
    let content = "<think>\nreasoning\n</think>\n\njawaban";
    let prompt = render(
        &[Turn {
            role: Role::Assistant,
            content,
        }],
        false,
    );

    assert!(prompt.contains("<|im_start|>assistant\n<think>\nreasoning\n</think>\n\njawaban"));
}

#[test]
fn thinking_disabled_closes_with_an_empty_block() {
    let prompt = render(
        &[Turn {
            role: Role::User,
            content: "Halo",
        }],
        false,
    );

    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
fn thinking_mode_leaves_the_block_open() {
    let prompt = render(
        &[Turn {
            role: Role::User,
            content: "Halo",
        }],
        true,
    );

    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
}
