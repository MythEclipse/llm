//! Compiles `kernels/gpu.cu` to PTX, embeds it, and records what was built.
//!
//! The GPU path is optional: without `nvcc` the crate still builds and
//! `Gpu::new()` reports that no device backend is available, so the CPU engine
//! keeps working on machines without a CUDA toolkit.
//!
//! Besides the PTX itself this writes the identity of the build — source hash,
//! PTX hash, architecture, compiler, kernel names — because every benchmark
//! number in the README has to name the artifact it came from.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const DEFAULT_ARCH: &str = "sm_89";

fn main() {
    println!("cargo:rerun-if-changed=kernels/gpu.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLM_CUDA_ARCH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let source = Path::new("kernels/gpu.cu");
    let arch = env::var("LLM_CUDA_ARCH").unwrap_or_else(|_| DEFAULT_ARCH.to_string());
    let ptx = out_dir.join("gpu.ptx");

    let source_hash = sha256_file(source).unwrap_or_default();
    let mut nvcc = "unavailable".to_string();
    let mut kernels: Vec<String> = Vec::new();
    let mut ptx_hash = String::new();

    let built = compile(source, &ptx, &arch).inspect(|()| {
        nvcc = compiler_version();
        kernels = entry_points(&ptx);
        ptx_hash = sha256_file(&ptx).unwrap_or_default();
    });
    if let Err(reason) = &built {
        println!("cargo:warning=GPU backend disabled: {reason}");
    }

    let ptx_binding = if built.is_ok() {
        format!("Some(include_str!({:?}))", ptx.to_string_lossy())
    } else {
        "None".to_string()
    };
    let ptx_hash_binding = if ptx_hash.is_empty() {
        "None".to_string()
    } else {
        format!("Some({ptx_hash:?})")
    };
    let generated = format!(
        "pub const PTX: Option<&str> = {ptx_binding};\n\
         pub const ARCH: &str = {arch:?};\n\
         pub const SOURCE_SHA256: &str = {source_hash:?};\n\
         pub const PTX_SHA256: Option<&str> = {ptx_hash_binding};\n\
         pub const NVCC: &str = {nvcc:?};\n\
         pub const KERNELS: &[&str] = &{kernels:?};\n"
    );
    std::fs::write(out_dir.join("gpu_build.rs"), generated).expect("write generated build info");
}

fn compile(source: &Path, ptx: &Path, arch: &str) -> Result<(), String> {
    let output = Command::new("nvcc")
        .args(["-ptx", "-arch", arch, "-O3", "-o"])
        .arg(ptx)
        .arg(source)
        .output()
        .map_err(|error| format!("nvcc not runnable ({error})"))?;

    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "nvcc failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

fn sha256_file(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

/// The `release 13.4, V13.4.92` line, without the surrounding noise.
fn compiler_version() -> String {
    let Ok(output) = Command::new("nvcc").arg("--version").output() else {
        return "unavailable".to_string();
    };
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .find_map(|line| line.trim().strip_prefix("Cuda compilation tools, "))
        .map(|line| line.split(',').next().unwrap_or(line).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Every `.entry <name>(` in the PTX, in emission order. The line reads
/// `.visible .entry name(` in nvcc's output.
fn entry_points(ptx: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(ptx) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|line| {
            let rest = line.split(".entry ").nth(1)?;
            let name = rest.split('(').next()?.trim();
            (!name.is_empty()).then(|| name.to_string())
        })
        .collect()
}
