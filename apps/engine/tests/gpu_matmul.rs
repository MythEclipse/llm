//! The GPU kernels have to produce the same numbers as the CPU kernels.
//!
//! Both compute `d · Σqᵢxᵢ` in f32, so only the summation order differs; the
//! budget below is what that ordering costs on logits of this magnitude.

use std::path::PathBuf;

use llm_engine::tensor::Weight;
use llm_engine::{Compute, DType, GgufFile, Model, Session};

fn model_path() -> PathBuf {
    std::env::var_os("LLM_MODEL").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/MiniCPM5-2B-Q8_0.gguf"),
        PathBuf::from,
    )
}

fn weight<'a>(gguf: &'a GgufFile, name: &str) -> Weight<'a> {
    let tensor = gguf.tensor(name).expect("tensor exists");
    let k = tensor.dims[0] as usize;
    let out = tensor.dims[1] as usize;
    assert_eq!(tensor.dtype, DType::Q8_0, "fixture must be quantized");
    Weight::new(tensor.dtype, k, out, gguf.data(name).expect("tensor data"))
}

fn activations(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|index| (index as f32 * 0.017 + seed).sin() * 0.5)
        .collect()
}

#[test]
#[ignore = "requires the model and a CUDA device"]
fn gpu_matmul_matches_the_cpu_kernel() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    // One weight matrix is enough to exercise the kernels.
    let gpu = Compute::auto(64 << 20);
    let Some(device) = gpu.device_name() else {
        println!("skipped: no CUDA device (LLM_GPU=0 or no driver)");
        return;
    };
    println!("device: {device}");

    let cpu = Compute::cpu();
    for name in ["blk.0.attn_q.weight", "blk.0.ffn_down.weight"] {
        let weight = weight(&gguf, name);
        for n in [1usize, 8, 64] {
            let x = activations(n * weight.k, n as f32);
            let mut expected = vec![0.0f32; n * weight.out];
            let mut actual = vec![0.0f32; n * weight.out];

            for (compute, result) in [(&cpu, &mut expected), (&gpu, &mut actual)] {
                let mut source = compute.alloc(n * weight.k).expect("alloc input");
                let mut target = compute.alloc(n * weight.out).expect("alloc output");
                compute.upload(&mut source, &x).expect("upload");
                compute
                    .matmul(&weight, &source, 0, n, &mut target, 0)
                    .expect("matmul");
                compute.download(&target, result).expect("download");
            }

            let max_delta = expected
                .iter()
                .zip(&actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = expected.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            println!("{name} n={n}: max |Δ| = {max_delta} (values up to {scale})");
            assert!(
                max_delta <= 1e-3 * scale.max(1.0),
                "{name} n={n}: max |Δ| = {max_delta}, scale {scale}"
            );
        }
    }
}

#[test]
#[ignore = "requires the model and a CUDA device"]
fn gpu_pipeline_keeps_the_greedy_first_token() {
    let gguf = GgufFile::open(&model_path()).expect("model opens");
    let model = Model::load(&gguf).expect("model loads");
    println!("device: {:?}", model.device_name());

    let tokens = [608u32, 4894, 304, 6918, 357];
    let mut session = Session::new(&model, 128).expect("session builds");
    let logits = session.prefill(&tokens).expect("prefill runs");
    let best = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index as u32);
    println!("argmax after prefill: {best:?}");
    assert_eq!(best, Some(8181), "greedy first token must stay ' Paris'");
}
