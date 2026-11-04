//! Compiles `kernels/gpu.cu` to PTX and embeds it.
//!
//! The GPU path is optional: without `nvcc` the crate still builds and
//! `Gpu::new()` reports that no device backend is available, so the CPU engine
//! keeps working on machines without a CUDA toolkit.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_ARCH: &str = "sm_89";

fn main() {
    println!("cargo:rerun-if-changed=kernels/gpu.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLM_CUDA_ARCH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let source = Path::new("kernels/gpu.cu");
    let arch = env::var("LLM_CUDA_ARCH").unwrap_or_else(|_| DEFAULT_ARCH.to_string());
    let ptx = out_dir.join("gpu.ptx");

    let generated = match compile(source, &ptx, &arch) {
        Ok(()) => format!(
            "pub const PTX: Option<&str> = Some(include_str!({:?}));\n",
            ptx.to_string_lossy()
        ),
        Err(reason) => {
            println!("cargo:warning=GPU backend disabled: {reason}");
            "pub const PTX: Option<&str> = None;\n".to_string()
        }
    };
    std::fs::write(out_dir.join("gpu_ptx.rs"), generated).expect("write generated PTX binding");
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
