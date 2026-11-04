//! CUDA backend: device buffers, weight residency, kernel launches.
//!
//! The whole forward pass runs here once a GPU is present — activations, KV
//! cache and weights all live in device memory. Staging activations through the
//! host per matmul costs a synchronisation round trip each time, which measured
//! slower than the CPU for single-token decode, so the boundary sits at the
//! model, not at the matmul.

mod sys;

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::Mutex;

use sys::{
    Context, Device, DevicePtr, Function, Module, SUCCESS, cuCtxCreate_v2, cuCtxDestroy_v2,
    cuCtxSetCurrent, cuDeviceGet, cuDeviceGetName, cuDeviceTotalMem_v2, cuInit, cuLaunchKernel,
    cuMemAlloc_v2, cuMemFree_v2, cuMemGetInfo_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
    cuModuleGetFunction, cuModuleLoadData, error_message,
};

use crate::error::EngineError;

include!(concat!(env!("OUT_DIR"), "/gpu_ptx.rs"));

const LANES: u32 = 32;
const WARPS: u32 = 8;
/// Four warps per attention block, matching `ATTENTION_WARPS` in the kernel.
const ATTENTION_THREADS: u32 = 128;
/// Output rows per block on the prompt path, matching `ROWS_PER_BLOCK`.
const ROWS_PER_BLOCK: usize = 4;

/// A device allocation of `len` f32 values, freed on drop.
pub struct DeviceBuffer {
    pointer: DevicePtr,
    len: usize,
}

impl DeviceBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `cuMemAlloc_v2` and is freed once.
        unsafe { cuMemFree_v2(self.pointer) };
    }
}

/// A kernel handle, looked up by name once at open time.
struct Kernels {
    matmul: Function,
    matmul_batch: Function,
    rmsnorm: Function,
    silu_mul: Function,
    residual_add: Function,
    rope: Function,
    kv_append: Function,
    attention: Function,
    embedding: Function,
}

/// A weight slice as the kernels want it.
enum Resident {
    /// Q8_0 repacked to `scales` (k/32 f32) + `quants` (k i8) per row.
    Quantized {
        scales: DeviceBuffer,
        quants: DeviceBuffer,
    },
    /// Anything else, uploaded verbatim.
    Plain(DeviceBuffer),
}

/// The device backend.
pub struct Gpu {
    context: Context,
    kernels: Kernels,
    name: String,
    total_bytes: usize,
    weights: Mutex<HashMap<(usize, usize), Resident>>,
}

impl Gpu {
    pub fn open() -> Result<Self, EngineError> {
        let Some(ptx) = PTX else {
            return Err(EngineError::Unsupported(
                "no GPU backend: this build had no CUDA compiler".into(),
            ));
        };

        // SAFETY: every driver call is checked; the handles live in `self`.
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut device = Device(0);
            check(cuDeviceGet(&raw mut device, 0), "cuDeviceGet")?;

            let mut name = [0i8; 128];
            check(
                cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, device),
                "cuDeviceGetName",
            )?;
            let name = std::ffi::CStr::from_ptr(name.as_ptr())
                .to_string_lossy()
                .into_owned();

            let mut total_bytes = 0usize;
            check(
                cuDeviceTotalMem_v2(&raw mut total_bytes, device),
                "cuDeviceTotalMem",
            )?;

            let mut context = Context(std::ptr::null_mut());
            check(cuCtxCreate_v2(&raw mut context, 0, device), "cuCtxCreate")?;

            let image = CString::new(ptx).expect("PTX has no interior NUL");
            let mut module = Module(std::ptr::null_mut());
            check(
                cuModuleLoadData(&raw mut module, image.as_ptr().cast::<c_void>()),
                "cuModuleLoadData",
            )?;

            let kernels = Kernels {
                matmul: function(module, "q8_matmul")?,
                matmul_batch: function(module, "q8_matmul_batch")?,
                rmsnorm: function(module, "rmsnorm_rows")?,
                silu_mul: function(module, "silu_mul")?,
                residual_add: function(module, "residual_add")?,
                rope: function(module, "rope")?,
                kv_append: function(module, "kv_append")?,
                attention: function(module, "attention")?,
                embedding: function(module, "embedding_row")?,
            };

            Ok(Self {
                context,
                kernels,
                name,
                total_bytes,
                weights: Mutex::new(HashMap::new()),
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Free device memory right now, used to decide whether a model fits.
    pub fn free_bytes(&self) -> Result<usize, EngineError> {
        self.bind()?;
        let mut free = 0usize;
        let mut total = 0usize;
        // SAFETY: both pointers are valid for one usize.
        check(
            unsafe { cuMemGetInfo_v2(&raw mut free, &raw mut total) },
            "cuMemGetInfo",
        )?;
        Ok(free)
    }

    pub fn alloc(&self, len: usize) -> Result<DeviceBuffer, EngineError> {
        self.bind()?;
        let mut pointer = DevicePtr(0);
        // SAFETY: the driver writes the allocation handle into `pointer`.
        let status = unsafe { cuMemAlloc_v2(&raw mut pointer, (len * 4).max(4)) };
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: allocation of {} bytes failed: {}",
                len * 4,
                error_message(status)
            )));
        }
        Ok(DeviceBuffer { pointer, len })
    }

    pub fn upload(&self, destination: &DeviceBuffer, source: &[f32]) -> Result<(), EngineError> {
        self.bind()?;
        assert!(source.len() <= destination.len, "upload larger than buffer");
        // SAFETY: both pointers are valid for the copied length.
        let status = unsafe {
            cuMemcpyHtoD_v2(
                destination.pointer,
                source.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(source),
            )
        };
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: host-to-device copy failed: {}",
                error_message(status)
            )));
        }
        Ok(())
    }

    pub fn download(
        &self,
        source: &DeviceBuffer,
        destination: &mut [f32],
    ) -> Result<(), EngineError> {
        assert!(
            destination.len() <= source.len,
            "download larger than buffer"
        );
        // SAFETY: both pointers are valid for the copied length.
        let status = unsafe {
            cuMemcpyDtoH_v2(
                destination.as_mut_ptr().cast::<c_void>(),
                source.pointer,
                std::mem::size_of_val(destination),
            )
        };
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: device-to-host copy failed: {}",
                error_message(status)
            )));
        }
        Ok(())
    }

    /// `y[t·out + o] = Σᵢ x[t·k + i] · W[o·k + i]`, weights uploaded once.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        identity: (usize, usize),
        weights: &[u8],
        x: &DeviceBuffer,
        x_offset: usize,
        n: usize,
        k: usize,
        out: usize,
        y: &DeviceBuffer,
        y_offset: usize,
    ) -> Result<(), EngineError> {
        assert!(
            k.is_multiple_of(32),
            "Q8_0 rows are whole 32-element blocks"
        );
        assert!(x.len() >= n * k, "activation shape mismatch");
        assert!(y.len() >= n * out, "output shape mismatch");

        let (scales, quants) = self.quantized(identity, weights)?;
        let mut args = [
            Arg::Pointer(scales),
            Arg::Pointer(quants),
            Arg::Pointer(shifted(x.pointer, x_offset)),
            Arg::Pointer(shifted(y.pointer, y_offset)),
            Arg::Int(k as i32),
            Arg::Int(out as i32),
            Arg::Int(n as i32),
        ];
        if n >= WARPS as usize {
            // Prompt path: four rows per block, eight tokens per warp, k chunked.
            return self.launch(
                self.kernels.matmul_batch,
                (out.div_ceil(ROWS_PER_BLOCK) as u32, 1, 1),
                (LANES, WARPS, 1),
                0,
                &mut args,
            );
        }
        // Decode path keeps the row in registers, so it needs no shared memory —
        // and asking for 24 KB anyway costs occupancy.
        self.launch(
            self.kernels.matmul,
            (out as u32, 1, 1),
            (LANES, WARPS, 1),
            0,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_rows(
        &self,
        source: &DeviceBuffer,
        source_offset: usize,
        destination: &DeviceBuffer,
        destination_offset: usize,
        weight: &[f32],
        n: usize,
        rows: usize,
        eps: f32,
    ) -> Result<(), EngineError> {
        let identity = (weight.as_ptr() as usize, weight.len());
        let device_weight = self.plain(identity, as_bytes(weight))?;
        let mut args = [
            Arg::Pointer(shifted(source.pointer, source_offset)),
            Arg::Pointer(shifted(destination.pointer, destination_offset)),
            Arg::Pointer(device_weight),
            Arg::Int(n as i32),
            Arg::Int(rows as i32),
            Arg::Float(eps),
        ];
        self.launch(
            self.kernels.rmsnorm,
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    pub fn silu_mul(
        &self,
        act: &DeviceBuffer,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngineError> {
        let mut args = [
            Arg::Pointer(act.pointer),
            Arg::Pointer(gate.pointer),
            Arg::Pointer(up.pointer),
            Arg::Int(n as i32),
        ];
        self.launch(
            self.kernels.silu_mul,
            (blocks(n), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    pub fn residual_add(
        &self,
        y: &DeviceBuffer,
        y_offset: usize,
        x: &DeviceBuffer,
        x_offset: usize,
        n: usize,
    ) -> Result<(), EngineError> {
        let mut args = [
            Arg::Pointer(shifted(y.pointer, y_offset)),
            Arg::Pointer(shifted(x.pointer, x_offset)),
            Arg::Int(n as i32),
        ];
        self.launch(
            self.kernels.residual_add,
            (blocks(n), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &DeviceBuffer,
        x_offset: usize,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        half: usize,
        n_rot: usize,
        base: f32,
        start: usize,
    ) -> Result<(), EngineError> {
        let total = tokens * heads * half;
        let mut args = [
            Arg::Pointer(shifted(x.pointer, x_offset)),
            Arg::Int(tokens as i32),
            Arg::Int(heads as i32),
            Arg::Int(head_dim as i32),
            Arg::Int(half as i32),
            Arg::Int(n_rot as i32),
            Arg::Float(base),
            Arg::Int(start as i32),
        ];
        self.launch(
            self.kernels.rope,
            (blocks(total), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(
        &self,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        source_keys: &DeviceBuffer,
        source_values: &DeviceBuffer,
        tokens: usize,
        kv_dim: usize,
        start: usize,
    ) -> Result<(), EngineError> {
        let total = tokens * kv_dim;
        let mut args = [
            Arg::Pointer(keys.pointer),
            Arg::Pointer(values.pointer),
            Arg::Pointer(source_keys.pointer),
            Arg::Pointer(source_values.pointer),
            Arg::Int(tokens as i32),
            Arg::Int(kv_dim as i32),
            Arg::Int(start as i32),
        ];
        self.launch(
            self.kernels.kv_append,
            (blocks(total), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        query: &DeviceBuffer,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        out: &DeviceBuffer,
        scores: &DeviceBuffer,
        geometry: AttentionGeometry,
        start: usize,
        tokens: usize,
    ) -> Result<(), EngineError> {
        let mut args = [
            Arg::Pointer(query.pointer),
            Arg::Pointer(keys.pointer),
            Arg::Pointer(values.pointer),
            Arg::Pointer(out.pointer),
            Arg::Pointer(scores.pointer),
            Arg::Int(geometry.n_head as i32),
            Arg::Int(geometry.n_head_kv as i32),
            Arg::Int(geometry.head_dim as i32),
            Arg::Int(geometry.scores_stride as i32),
            Arg::Int(start as i32),
            Arg::Int(tokens as i32),
            Arg::Float(geometry.scale),
        ];
        self.launch(
            self.kernels.attention,
            (geometry.n_head as u32, tokens as u32, 1),
            (ATTENTION_THREADS, 1, 1),
            0,
            &mut args,
        )
    }

    pub fn embedding(
        &self,
        identity: (usize, usize),
        weights: &[u8],
        row: u32,
        out: &DeviceBuffer,
        out_offset: usize,
        k: usize,
    ) -> Result<(), EngineError> {
        let (scales, quants) = self.quantized(identity, weights)?;
        let mut args = [
            Arg::Pointer(scales),
            Arg::Pointer(quants),
            Arg::Pointer(shifted(out.pointer, out_offset)),
            Arg::Int(row as i32),
            Arg::Int(k as i32),
        ];
        self.launch(
            self.kernels.embedding,
            (blocks(k), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    /// Uploads a weight slice now, failing early if the device is full.
    pub fn reserve(&self, identity: (usize, usize), bytes: &[u8]) -> Result<(), EngineError> {
        self.quantized(identity, bytes).map(|_| ())
    }

    /// Makes this context current on the calling thread.
    fn bind(&self) -> Result<(), EngineError> {
        // SAFETY: the context was created by `open` and is alive for `self`.
        check(unsafe { cuCtxSetCurrent(self.context) }, "cuCtxSetCurrent")
    }

    /// Uploads a Q8_0 slice once, repacked, and returns the two device pointers.
    fn quantized(
        &self,
        identity: (usize, usize),
        bytes: &[u8],
    ) -> Result<(DevicePtr, DevicePtr), EngineError> {
        let mut cache = self.weights.lock().expect("weight cache lock");
        if let Some(Resident::Quantized { scales, quants }) = cache.get(&identity) {
            return Ok((scales.pointer, quants.pointer));
        }
        let (scales, quants) = repack_q8(bytes);
        let device_scales = self.upload_new(&scales)?;
        let device_quants = self.upload_new_bytes(as_bytes(&quants))?;
        let pointers = (device_scales.pointer, device_quants.pointer);
        cache.insert(
            identity,
            Resident::Quantized {
                scales: device_scales,
                quants: device_quants,
            },
        );
        Ok(pointers)
    }

    /// Uploads a non-quantized slice (norm weights) verbatim.
    fn plain(&self, identity: (usize, usize), bytes: &[u8]) -> Result<DevicePtr, EngineError> {
        let mut cache = self.weights.lock().expect("weight cache lock");
        if let Some(Resident::Plain(buffer)) = cache.get(&identity) {
            return Ok(buffer.pointer);
        }
        let buffer = self.upload_new_bytes(bytes)?;
        let pointer = buffer.pointer;
        cache.insert(identity, Resident::Plain(buffer));
        Ok(pointer)
    }

    fn upload_new(&self, values: &[f32]) -> Result<DeviceBuffer, EngineError> {
        self.upload_new_bytes(as_bytes(values))
    }

    fn upload_new_bytes(&self, bytes: &[u8]) -> Result<DeviceBuffer, EngineError> {
        self.bind()?;
        let buffer = self.alloc(bytes.len().div_ceil(4))?;
        // SAFETY: the buffer holds at least `bytes.len()` bytes.
        let status = unsafe {
            cuMemcpyHtoD_v2(buffer.pointer, bytes.as_ptr().cast::<c_void>(), bytes.len())
        };
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: weight upload failed: {}",
                error_message(status)
            )));
        }
        Ok(buffer)
    }

    fn launch(
        &self,
        function: Function,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        args: &mut [Arg],
    ) -> Result<(), EngineError> {
        self.bind()?;
        let mut pointers: Vec<*mut c_void> = args
            .iter_mut()
            .map(|arg| match arg {
                Arg::Pointer(value) => std::ptr::from_mut(value).cast::<c_void>(),
                Arg::Int(value) => std::ptr::from_mut(value).cast::<c_void>(),
                Arg::Float(value) => std::ptr::from_mut(value).cast::<c_void>(),
            })
            .collect();
        // SAFETY: `args` matches the kernel signature and outlives the call.
        let status = unsafe {
            cuLaunchKernel(
                function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_bytes,
                std::ptr::null_mut(),
                pointers.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: kernel launch failed: {}",
                error_message(status)
            )));
        }
        Ok(())
    }
}

// SAFETY: every driver call binds the context to the calling thread first, and
// the engine runs one generation at a time, so the handles are never used
// concurrently. The kernel launches are serialised by the driver's own stream.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.weights.lock().expect("weight cache lock").clear();
        // SAFETY: the context was created by this struct and is destroyed once.
        unsafe {
            let _ = cuCtxDestroy_v2(self.context);
        }
    }
}

/// One kernel parameter: device pointers, integers, floats, in declaration order.
enum Arg {
    Pointer(DevicePtr),
    Int(i32),
    Float(f32),
}

/// Shape of the attention kernel's grid, kept in one place.
pub struct AttentionGeometry {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub scores_stride: usize,
    pub scale: f32,
}

fn function(module: Module, name: &str) -> Result<Function, EngineError> {
    let symbol = CString::new(name).expect("kernel name is static");
    let mut function = Function(std::ptr::null_mut());
    // SAFETY: the module is loaded and the symbol name is NUL-terminated.
    check(
        unsafe { cuModuleGetFunction(&raw mut function, module, symbol.as_ptr()) },
        "cuModuleGetFunction",
    )?;
    Ok(function)
}

/// Splits GGUF Q8_0 rows into a scale plane and a quant plane.
///
/// GGUF stores `[f16 scale][32 i8]` per 32 elements, so a row is a stride-34
/// walk. Splitting it lets the kernel load eight quants in one instruction.
fn repack_q8(bytes: &[u8]) -> (Vec<f32>, Vec<i8>) {
    const BLOCK: usize = 34;
    let blocks = bytes.len() / BLOCK;
    let mut scales = Vec::with_capacity(blocks);
    let mut quants = Vec::with_capacity(blocks * 32);
    for block in bytes.as_chunks::<BLOCK>().0 {
        let bits = u16::from_le_bytes([block[0], block[1]]);
        scales.push(half::f16::from_bits(bits).to_f32());
        quants.extend(block[2..].iter().map(|byte| *byte as i8));
    }
    (scales, quants)
}

/// Device pointer moved by `floats` elements.
fn shifted(pointer: DevicePtr, floats: usize) -> DevicePtr {
    DevicePtr(pointer.0 + (floats as u64) * 4)
}

fn blocks(elements: usize) -> u32 {
    (elements.div_ceil(256)).clamp(1, 65_535) as u32
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: f32 has no padding and every bit pattern is valid.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn check(status: i32, operation: &str) -> Result<(), EngineError> {
    if status == SUCCESS {
        return Ok(());
    }
    Err(EngineError::Unsupported(format!(
        "cuda: {operation} failed: {}",
        error_message(status)
    )))
}
