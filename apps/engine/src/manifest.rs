//! What this binary was built from.
//!
//! A performance number is only evidence if it names the code it came from, so
//! the build writes the identity of the kernel source into the binary: the
//! architecture, the hashes of the source and of the PTX that was actually
//! compiled, the compiler that produced it, and the kernel entry points.
//!
//! Two builds with the same source, arch and compiler hash to the same value; a
//! changed kernel or a changed arch does not.

include!(concat!(env!("OUT_DIR"), "/gpu_build.rs"));

use sha2::{Digest, Sha256};

/// Identity of the compiled artifact.
#[derive(Debug, Clone)]
pub struct BuildManifest {
    pub arch: &'static str,
    /// Hash of `kernels/gpu.cu` as it was when this binary was built.
    pub source_sha256: &'static str,
    /// Hash of the PTX nvcc produced, `None` when the build had no compiler.
    pub ptx_sha256: Option<&'static str>,
    pub nvcc: &'static str,
    pub kernels: &'static [&'static str],
}

impl BuildManifest {
    pub fn current() -> Self {
        Self {
            arch: ARCH,
            source_sha256: SOURCE_SHA256,
            ptx_sha256: PTX_SHA256,
            nvcc: NVCC,
            kernels: KERNELS,
        }
    }

    /// Whether this build can run kernels at all.
    pub fn has_backend(&self) -> bool {
        self.ptx_sha256.is_some()
    }

    /// The artifact identity: everything that decides what machine code runs.
    ///
    /// Kept separate from the kernel *selection* (see [`crate::tactics`]), which
    /// can differ between two runs of the same binary on different hardware.
    pub fn identity(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.arch.as_bytes());
        hasher.update(b"\n");
        hasher.update(self.source_sha256.as_bytes());
        hasher.update(b"\n");
        hasher.update(self.ptx_sha256.unwrap_or("none").as_bytes());
        hasher.update(b"\n");
        for kernel in self.kernels {
            hasher.update(kernel.as_bytes());
            hasher.update(b"\n");
        }
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_a_full_digest_and_stable() {
        let manifest = BuildManifest::current();
        let identity = manifest.identity();
        assert_eq!(identity.len(), 64);
        assert_eq!(identity, manifest.identity());
        assert_eq!(manifest.source_sha256.len(), 64);
    }

    #[test]
    fn a_ptx_build_names_its_kernels() {
        let manifest = BuildManifest::current();
        if manifest.has_backend() {
            assert!(manifest.kernels.len() > 10, "{:?}", manifest.kernels);
            assert!(manifest.kernels.contains(&"q8_matmul_mma"));
        }
    }
}
