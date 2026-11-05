//! CUDA backend: device buffers, weight residency, kernel launches.
//!
//! The whole forward pass runs here once a GPU is present — activations, KV
//! cache and weights all live in device memory. Staging activations through the
//! host per matmul costs a synchronisation round trip each time, which measured
//! slower than the CPU for single-token decode, so the boundary sits at the
//! model, not at the matmul.

mod sys;

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Arc, Mutex};

use std::sync::atomic::{AtomicUsize, Ordering};

use half::f16;

use sys::{
    ATTRIBUTE_MULTIPROCESSORS, Context, Device, DevicePtr, Event,
    FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED, FUNC_ATTRIBUTE_MAX_THREADS, FUNC_ATTRIBUTE_NUM_REGS,
    FUNC_ATTRIBUTE_SHARED_SIZE, Function, GraphExec, Module, STREAM_CAPTURE_THREAD_LOCAL, SUCCESS,
    Stream, cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxSetCurrent, cuDeviceGet, cuDeviceGetAttribute,
    cuDeviceGetName, cuDeviceTotalMem_v2, cuEventCreate, cuEventDestroy, cuEventElapsedTime,
    cuEventRecord, cuEventSynchronize, cuFuncGetAttribute, cuFuncGetParamCount, cuFuncSetAttribute,
    cuGraphDestroy, cuGraphExecDestroy, cuGraphInstantiate_v2, cuGraphLaunch, cuInit,
    cuLaunchKernel, cuMemAlloc_v2, cuMemFree_v2, cuMemGetInfo_v2, cuMemcpyDtoHAsync_v2,
    cuMemcpyHtoDAsync_v2, cuModuleGetFunction, cuModuleLoadData,
    cuOccupancyMaxActiveBlocksPerMultiprocessor, cuStreamBeginCapture, cuStreamCreate,
    cuStreamDestroy_v2, cuStreamEndCapture, cuStreamSynchronize, error_message,
};

use crate::error::EngineError;
use tracing::{debug, info, trace};

/// Bytes the engine holds on the device, and the high-water mark of that.
/// Process-wide, because every buffer is born in `Gpu::alloc` and frees itself on
/// drop: the two counters then describe the engine's own footprint, the number
/// llama.cpp prints as its model and KV buffer sizes, rather than the driver's
/// view of the whole context.
static DEVICE_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEVICE_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Bytes one allocation of `len` f32 values reserves, matching what `Gpu::alloc`
/// asks the driver for and what `DeviceBuffer::drop` gives back.
fn allocation_bytes(len: usize) -> usize {
    (len * 4).max(4)
}

fn note_allocation(bytes: usize) {
    let live = DEVICE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    DEVICE_PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn note_release(bytes: usize) {
    DEVICE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

use crate::manifest::PTX;

const LANES: u32 = 32;
/// Matrices one prompt matmul launch carries, matching `MMA_GROUP_MAX`.
pub const MMA_GROUP_MAX: usize = 4;
const WARPS: u32 = 8;
/// Output rows one decode matmul block owns, matching `DECODE_ROWS_PER_BLOCK`.
const DECODE_ROWS_PER_BLOCK: usize = 4;
/// Four warps per attention block, matching `ATTENTION_WARPS` in the kernel.
const ATTENTION_THREADS: u32 = 128;
/// Positions per attention block on the decode path, and the shared window the
/// kernel reserves per block.
const SPLIT_POSITIONS: usize = 16;
const MAX_SPLIT_POSITIONS: usize = 256;
const MAX_SPLITS: usize = 1024;

/// Enough blocks to fill the device, and never a window wider than the kernel's
/// shared array — the second bound is what keeps long contexts legal.
///
/// This is also the shape key of a captured decode step: two calls with the same
/// split count launch the same grid, so one graph serves both.
pub fn split_count(visible: usize) -> usize {
    let wanted = visible.div_ceil(SPLIT_POSITIONS).max(1);
    let required = visible.div_ceil(MAX_SPLIT_POSITIONS).max(1);
    wanted.max(required).min(MAX_SPLITS)
}

/// Query positions per prefill attention block, matching `FA_QUERY_TILE`.
const PREFILL_QUERY_TILE: usize = 16;
/// Heads per prefill attention block, matching `FA_HEADS`: four heads that share
/// one KV head, so the staged keys and values are read once instead of four times.
const PREFILL_HEADS: usize = 2;
/// Warps per prefill attention block, matching `FA_WARPS`.
const PREFILL_WARPS: usize = 8;
/// Shared memory the tiled attention kernel asks for: four heads of queries, one
/// key tile, one value tile, and the probabilities.
/// Shared memory the tiled attention kernel asks for: two heads of queries, one key
/// tile and one value tile. The probabilities stay in registers, which is what keeps
/// the block small enough for two of them to share an SM.
const PREFILL_SHARED_BYTES: u32 = 49_152;

/// Tensor-core tile on the prompt path: sixteen rows per warp, four warps.
const MMA_ROWS: usize = 16;
/// Sixteen warps of sixteen rows: a block owns 256 output rows, which halves how
/// often the activation tile is staged, and the SM has twice as many warps to
/// hide the staging latency with.
const MMA_WARPS: usize = 16;
/// Dynamic shared memory the prompt matmul asks for: two k-chunk stages of
/// weights and activations, both padded by `MMA_PAD`, plus their per-block scales.
/// 2 × 128 × 144 twice over, plus 2 × 2 × 128 × 4 scales.
const MMA_SHARED_BYTES: u32 = 77_824;
/// Tokens one prompt block can hold, matching `MMA_BLOCK_TOKENS` in the kernel
/// and the engine's prefill chunk.
/// Tokens one prompt-matmul block covers, matching `MMA_BLOCK_TOKENS`.
const MMA_BLOCK_TOKENS: usize = 128;
/// Largest prompt the kernel is launched with, matching `PREFILL_CHUNK`.
const MMA_TOKENS: usize = 1024;

/// A device allocation of `len` f32 values, freed on drop.
/// Activations already quantised on the device, ready for one or more matmuls.
pub struct Prepared(DevicePtr);

/// One device allocation several [`DeviceBuffer`]s carve up, freed when the
/// last view drops. The scratch plan owns one of these per session.
pub struct Arena {
    pointer: DevicePtr,
    bytes: usize,
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `cuMemAlloc_v2` and is freed once.
        unsafe { cuMemFree_v2(self.pointer) };
        note_release(self.bytes);
    }
}

pub struct DeviceBuffer {
    pointer: DevicePtr,
    len: usize,
    /// Set when this buffer is a window into an [`Arena`], which owns the
    /// allocation and frees it when the last view goes.
    arena: Option<Arc<Arena>>,
}

impl DeviceBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer, for kernels that read one buffer and add it into
    /// another.
    pub fn pointer(&self) -> DevicePtr {
        self.pointer
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.arena.is_some() {
            return;
        }
        // SAFETY: the pointer came from `cuMemAlloc_v2` and is freed once.
        unsafe { cuMemFree_v2(self.pointer) };
        note_release(allocation_bytes(self.len));
    }
}

/// A `len`-float window into `arena`, starting `offset` floats in. The view does
/// not free: the arena does, once the last view is gone.
pub fn view(arena: &Arc<Arena>, offset: usize, len: usize) -> DeviceBuffer {
    // The caller only asks for windows its own plan placed inside the arena, and
    // the arena outlives the view through the `Arc`.
    let pointer = shifted(arena.pointer, offset);
    DeviceBuffer {
        pointer,
        len,
        arena: Some(Arc::clone(arena)),
    }
}

/// `[position, visible]` on the device. A captured decode graph bakes kernel
/// arguments, so the values that change per token live in memory the kernels
/// read instead.
///
/// `cuStreamWriteValue32` would write it without a host copy, but this device
/// reports `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 0`, so the words go
/// over as one eight-byte asynchronous copy instead.
struct StateBuffer {
    buffer: DeviceBuffer,
    /// Host mirror the copy reads from; it has to outlive the copy.
    values: Box<[i32; 2]>,
}

/// A captured decode step: the launches are recorded once and replayed with one
/// driver call, on the same pointers.
pub struct Graph {
    exec: GraphExec,
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: the handle came from `cuGraphInstantiate_v2` and is destroyed once.
        unsafe {
            let _ = cuGraphExecDestroy(self.exec);
        }
    }
}

/// What one kernel actually costs on this device, read from the driver instead
/// of assumed: registers, static shared memory, the largest legal block, and how
/// many blocks of the launched shape fit on one multiprocessor.
///
/// The wave plan and the shared-memory ceilings used to be numbers someone had
/// to keep true by hand. They are now read once, at load, and the launch shapes
/// are checked against them there — the launch path assumes nothing.
#[derive(Debug, Clone, Copy)]
pub struct KernelResources {
    pub name: &'static str,
    pub registers: i32,
    pub static_shared: i32,
    pub max_threads: i32,
    pub block_threads: u32,
    pub dynamic_shared: u32,
    pub blocks_per_sm: i32,
}

/// Reads one kernel's cost and rejects a launch shape it cannot take.
fn inspect(
    function: Function,
    name: &'static str,
    block: (u32, u32, u32),
    dynamic_shared: u32,
) -> Result<KernelResources, EngineError> {
    let threads = block.0 * block.1 * block.2;
    let mut max_threads = 0i32;
    let mut static_shared = 0i32;
    let mut registers = 0i32;
    // SAFETY: `function` is a live handle and each call writes one `i32`.
    unsafe {
        check(
            cuFuncGetAttribute(&raw mut max_threads, FUNC_ATTRIBUTE_MAX_THREADS, function),
            "cuFuncGetAttribute(max threads)",
        )?;
        check(
            cuFuncGetAttribute(&raw mut static_shared, FUNC_ATTRIBUTE_SHARED_SIZE, function),
            "cuFuncGetAttribute(shared)",
        )?;
        check(
            cuFuncGetAttribute(&raw mut registers, FUNC_ATTRIBUTE_NUM_REGS, function),
            "cuFuncGetAttribute(registers)",
        )?;
    }
    if threads as i32 > max_threads {
        return Err(EngineError::Unsupported(format!(
            "cuda: {name} launches {threads} threads, its maximum is {max_threads}"
        )));
    }
    let mut blocks = 0i32;
    // SAFETY: the handle is live and the driver writes one `i32`.
    unsafe {
        check(
            cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &raw mut blocks,
                function,
                threads as i32,
                dynamic_shared as usize,
            ),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )?;
    }
    Ok(KernelResources {
        name,
        registers,
        static_shared,
        max_threads,
        block_threads: threads,
        dynamic_shared,
        blocks_per_sm: blocks.max(1),
    })
}

/// A kernel handle, looked up by name once at open time.
#[allow(dead_code)]
struct Kernels {
    matmul: Function,
    matmul_decode: Function,
    matmul_mma: Function,
    quantize_rows: Function,
    silu_mul_quantize: Function,
    rmsnorm: Function,
    rmsnorm_quantize: Function,
    silu_mul: Function,
    split_query_gate: Function,
    sigmoid_mul: Function,
    copy_f32: Function,
    fill_f32: Function,
    axpby_f32: Function,
    rope: Function,
    rope_append: Function,
    kv_append: Function,
    attention: Function,
    attention_prefill: Function,
    attention_prefill_mma: Function,
    attention_split: Function,
    attention_combine: Function,
    embedding: Function,
    embedding_rows: Function,
}

/// One matmul of a group that shares its activations.
pub struct GroupEntry<'a> {
    pub identity: (usize, usize),
    pub bytes: &'a [u8],
    pub out: usize,
    pub y: &'a DeviceBuffer,
    pub y_offset: usize,
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
    /// Streaming multiprocessors.
    multiprocessors: usize,
    /// Blocks the prompt matmul puts on one multiprocessor, read from the driver.
    prompt_blocks_per_sm: usize,
    /// One stream for every launch and copy, so the order is the one written
    /// here and a decode step can be captured.
    stream: Stream,
    weights: Mutex<HashMap<(usize, usize), Resident>>,
    workspace: Mutex<Option<DeviceBuffer>>,
    activation: Mutex<Option<DeviceBuffer>>,
    tokens: Mutex<Option<DeviceBuffer>>,
    /// `[position, visible]` on the device, so a captured graph reads the
    /// position of the token it is replaying instead of the one it was
    /// captured with.
    state: Mutex<Option<StateBuffer>>,
    names: Vec<(Function, &'static str)>,
    /// Parameters each kernel declares. A launch that passes a different number
    /// would be handing the driver a buffer it will read past.
    params: Vec<(Function, i32)>,
    profile_enabled: bool,
    profile: Mutex<Option<HashMap<&'static str, (f64, u64)>>>,
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

            let mut multiprocessors = 0i32;
            check(
                cuDeviceGetAttribute(&raw mut multiprocessors, ATTRIBUTE_MULTIPROCESSORS, device),
                "cuDeviceGetAttribute",
            )?;

            let mut context = Context(std::ptr::null_mut());
            check(cuCtxCreate_v2(&raw mut context, 0, device), "cuCtxCreate")?;

            // One stream for the whole engine: capture needs a stream that is not
            // the legacy default one, and a single order keeps every copy where
            // the caller put it.
            let mut stream: Stream = std::ptr::null_mut();
            check(cuStreamCreate(&raw mut stream, 0), "cuStreamCreate")?;

            let image = CString::new(ptx).expect("PTX has no interior NUL");
            let mut module = Module(std::ptr::null_mut());
            check(
                cuModuleLoadData(&raw mut module, image.as_ptr().cast::<c_void>()),
                "cuModuleLoadData",
            )?;

            let mut names: Vec<(Function, &'static str)> = Vec::new();
            let mut load = |name: &'static str| -> Result<Function, EngineError> {
                let loaded = function(module, name)?;
                names.push((loaded, name));
                Ok(loaded)
            };
            let kernels = Kernels {
                matmul: load("q8_matmul")?,
                matmul_decode: load("q8_matmul_decode")?,
                // The prompt matmul stages a 256-element k chunk, which is 57 KB of
                // shared memory, so its ceiling is raised once here.
                matmul_mma: reserve_shared(load("q8_matmul_mma")?, MMA_SHARED_BYTES)?,
                quantize_rows: load("quantize_rows")?,
                silu_mul_quantize: load("silu_mul_quantize_rows")?,
                rmsnorm: load("rmsnorm_rows")?,
                rmsnorm_quantize: load("rmsnorm_quantize_rows")?,
                silu_mul: load("silu_mul")?,
                split_query_gate: load("split_query_gate")?,
                sigmoid_mul: load("sigmoid_mul")?,
                copy_f32: load("copy_f32")?,
                fill_f32: load("fill_f32")?,
                axpby_f32: load("axpby_f32")?,
                rope: load("rope")?,
                rope_append: load("rope_append")?,
                kv_append: load("kv_append")?,
                attention: load("attention")?,
                attention_prefill: reserve_shared(
                    load("attention_prefill")?,
                    PREFILL_SHARED_BYTES,
                )?,
                attention_prefill_mma: reserve_shared(
                    load("attention_prefill_mma")?,
                    PREFILL_SHARED_BYTES,
                )?,
                attention_split: load("attention_split")?,
                attention_combine: load("attention_combine")?,
                embedding: load("embedding_row")?,
                embedding_rows: load("embedding_rows")?,
            };

            // What the kernels cost, and whether the shapes they are launched
            // with are legal for this device. Three of them decide the rest:
            // the prompt matmul (77 KB of shared, sixteen warps), the tiled
            // prefill attention (48 KB, eight warps) and the decode matmul
            // (no shared, four rows per block).
            let resources = [
                inspect(
                    kernels.matmul_mma,
                    "q8_matmul_mma",
                    (LANES, MMA_WARPS as u32, 1),
                    MMA_SHARED_BYTES,
                )?,
                inspect(
                    kernels.attention_prefill_mma,
                    "attention_prefill_mma",
                    (LANES, 8, 1),
                    PREFILL_SHARED_BYTES,
                )?,
                inspect(
                    kernels.matmul_decode,
                    "q8_matmul_decode",
                    ((DECODE_ROWS_PER_BLOCK as u32) * LANES, 1, 1),
                    0,
                )?,
            ];
            for resource in &resources {
                info!(
                    kernel = resource.name,
                    registers = resource.registers,
                    static_shared = resource.static_shared,
                    dynamic_shared = resource.dynamic_shared,
                    threads = resource.block_threads,
                    max_threads = resource.max_threads,
                    blocks_per_sm = resource.blocks_per_sm,
                    "kernel resources"
                );
            }
            let prompt_blocks_per_sm = resources[0].blocks_per_sm as usize;
            let mut params: Vec<(Function, i32)> = Vec::with_capacity(names.len());
            for (function, name) in &names {
                let mut count = 0usize;
                // SAFETY: the handle is live and the driver writes one `usize`.
                check(
                    cuFuncGetParamCount(*function, &raw mut count),
                    "cuFuncGetParamCount",
                )?;
                trace!(kernel = name, parameters = count, "kernel signature");
                params.push((*function, count as i32));
            }

            Ok(Self {
                names,
                params,
                profile_enabled: std::env::var("LLM_PROFILE").is_ok(),
                profile: Mutex::new(std::env::var("LLM_PROFILE").ok().map(|_| HashMap::new())),
                context,
                kernels,
                name,
                total_bytes,
                multiprocessors: multiprocessors.max(1) as usize,
                prompt_blocks_per_sm,
                weights: Mutex::new(HashMap::new()),
                stream,
                workspace: Mutex::new(None),
                activation: Mutex::new(None),
                tokens: Mutex::new(None),
                state: Mutex::new(None),
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Waits for every launch queued on this engine's stream.
    pub fn synchronize(&self) {
        // SAFETY: the context is current on this thread, set by each entry point.
        unsafe {
            cuCtxSetCurrent(self.context);
            let _ = cuStreamSynchronize(self.stream);
        }
    }

    /// Host to device, waiting for it. The asynchronous call plus a stream wait
    /// is the synchronous copy it replaces, but ordered against the launches
    /// instead of against a stream this engine does not use.
    fn copy_to_device(&self, destination: DevicePtr, source: *const c_void, bytes: usize) -> i32 {
        // SAFETY: both pointers are valid for `bytes`; the stream belongs here.
        unsafe {
            let status = cuMemcpyHtoDAsync_v2(destination, source, bytes, self.stream);
            if status == SUCCESS {
                cuStreamSynchronize(self.stream)
            } else {
                status
            }
        }
    }

    /// Device to host, waiting for it.
    fn copy_from_device(&self, destination: *mut c_void, source: DevicePtr, bytes: usize) -> i32 {
        // SAFETY: both pointers are valid for `bytes`; the stream belongs here.
        unsafe {
            let status = cuMemcpyDtoHAsync_v2(destination, source, bytes, self.stream);
            if status == SUCCESS {
                cuStreamSynchronize(self.stream)
            } else {
                status
            }
        }
    }

    /// Blocks that run concurrently: one wave.
    /// Blocks one wave holds for the prompt matmul: multiprocessors times what
    /// the driver says fits on each. The wave plan used to assume one.
    pub fn blocks_per_wave(&self) -> usize {
        self.multiprocessors * self.prompt_blocks_per_sm
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
        let status = unsafe { cuMemAlloc_v2(&raw mut pointer, allocation_bytes(len)) };
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: allocation of {} bytes failed: {}",
                len * 4,
                error_message(status)
            )));
        }
        note_allocation(allocation_bytes(len));
        Ok(DeviceBuffer {
            pointer,
            len,
            arena: None,
        })
    }

    /// One allocation the scratch buffers carve up, so a token allocates
    /// nothing after the session starts.
    pub fn arena(&self, floats: usize) -> Result<Arc<Arena>, EngineError> {
        self.bind()?;
        let bytes = allocation_bytes(floats);
        let mut pointer = DevicePtr(0);
        // SAFETY: the driver writes the allocation handle into `pointer`.
        let status = unsafe { cuMemAlloc_v2(&raw mut pointer, bytes) };
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: arena allocation of {bytes} bytes failed: {}",
                error_message(status)
            )));
        }
        note_allocation(bytes);
        Ok(Arc::new(Arena { pointer, bytes }))
    }

    /// Writes the position this pass works at, and the number of visible keys,
    /// where the kernels read them. A captured graph bakes arguments, so these
    /// two values cannot be arguments.
    pub fn set_position(&self, position: usize, visible: usize) -> Result<(), EngineError> {
        self.bind()?;
        let mut slot = self.state.lock().expect("state lock");
        if slot.is_none() {
            *slot = Some(StateBuffer {
                buffer: self.alloc(2)?,
                values: Box::new([0, 0]),
            });
        }
        let state = slot.as_mut().expect("just created");
        state.values[0] = position as i32;
        state.values[1] = visible as i32;
        // Asynchronous on purpose: the stream orders this copy before the
        // launches that read it, so a decode step does not wait on the host. The
        // source lives in this struct, so it outlives the copy.
        // SAFETY: both pointers are valid for the copied length.
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(
                state.buffer.pointer,
                state.values.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(&*state.values),
                self.stream,
            )
        };
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: position upload failed: {}",
                error_message(status)
            )));
        }
        Ok(())
    }

    /// `[position, visible]` as the kernels receive it.
    fn state_pointers(&self) -> Result<(DevicePtr, DevicePtr), EngineError> {
        let slot = self.state.lock().expect("state lock");
        let state = slot
            .as_ref()
            .ok_or_else(|| EngineError::Unsupported("cuda: position was never set".into()))?;
        Ok((state.buffer.pointer, shifted(state.buffer.pointer, 1)))
    }

    /// Starts recording launches on the stream. Everything recorded must be a
    /// launch on a pointer that will not move, and nothing may allocate while it
    /// is open.
    pub fn begin_capture(&self) -> Result<(), EngineError> {
        self.bind()?;
        if self.profile_enabled {
            return Err(EngineError::Unsupported(
                "cuda: profiling records events, which cannot be captured".into(),
            ));
        }
        // SAFETY: the stream belongs to this engine and is not capturing yet.
        check(
            unsafe { cuStreamBeginCapture(self.stream, STREAM_CAPTURE_THREAD_LOCAL) },
            "cuStreamBeginCapture",
        )
    }

    /// Closes the recording and instantiates it, ready to replay.
    pub fn end_capture(&self) -> Result<Graph, EngineError> {
        let mut graph = std::ptr::null_mut();
        // SAFETY: a capture is open on this stream; the driver writes one handle.
        check(
            unsafe { cuStreamEndCapture(self.stream, &raw mut graph) },
            "cuStreamEndCapture",
        )?;
        let mut exec = std::ptr::null_mut();
        let mut error_node = std::ptr::null_mut();
        let mut message = [0 as c_char; 256];
        // SAFETY: `graph` is a live handle; the buffers are ours.
        let status = unsafe {
            cuGraphInstantiate_v2(
                &raw mut exec,
                graph,
                &raw mut error_node,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        // SAFETY: the graph is not needed once instantiated.
        unsafe {
            let _ = cuGraphDestroy(graph);
        }
        if status != SUCCESS {
            let detail: String = message
                .iter()
                .take_while(|byte| **byte != 0)
                .map(|byte| *byte as u8 as char)
                .collect();
            return Err(EngineError::Unsupported(format!(
                "cuda: graph instantiation failed: {} {detail}",
                error_message(status)
            )));
        }
        Ok(Graph { exec })
    }

    /// Abandons a capture that failed halfway, so the stream is usable again.
    pub fn cancel_capture(&self) {
        let mut graph = std::ptr::null_mut();
        // SAFETY: ending the capture is the only way back; the graph is dropped.
        unsafe {
            if cuStreamEndCapture(self.stream, &raw mut graph) == SUCCESS && !graph.is_null() {
                let _ = cuGraphDestroy(graph);
            }
        }
    }

    /// Replays a captured step in one call.
    pub fn replay(&self, graph: &Graph) -> Result<(), EngineError> {
        self.bind()?;
        // SAFETY: the exec was instantiated by this engine and is alive.
        check(
            unsafe { cuGraphLaunch(graph.exec, self.stream) },
            "cuGraphLaunch",
        )
    }

    /// Bytes held on the device right now, and the most ever held at once.
    pub fn device_bytes(&self) -> (usize, usize) {
        (
            DEVICE_BYTES.load(Ordering::Relaxed),
            DEVICE_PEAK_BYTES.load(Ordering::Relaxed),
        )
    }

    pub fn upload(&self, destination: &DeviceBuffer, source: &[f32]) -> Result<(), EngineError> {
        self.bind()?;
        assert!(source.len() <= destination.len, "upload larger than buffer");
        // SAFETY: both pointers are valid for the copied length.
        let status = self.copy_to_device(
            destination.pointer,
            source.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(source),
        );
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
        let status = self.copy_from_device(
            destination.as_mut_ptr().cast::<c_void>(),
            source.pointer,
            std::mem::size_of_val(destination),
        );
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: device-to-host copy failed: {}",
                error_message(status)
            )));
        }
        Ok(())
    }

    /// `y[t·out + o] = Σᵢ x[t·k + i] · W[o·k + i]`, weights uploaded once.
    /// When `bias` is set, the result is added into `y` instead of overwriting
    /// it — the residual add happens in the epilogue, not in a separate pass.
    #[allow(clippy::too_many_arguments)]
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
        prepared: Option<&Prepared>,
        bias: Option<(DevicePtr, usize)>,
    ) -> Result<(), EngineError> {
        assert!(
            k.is_multiple_of(32),
            "Q8_0 rows are whole 32-element blocks"
        );
        assert!(x.len() >= n * k, "activation shape mismatch");
        assert!(y.len() >= n * out, "output shape mismatch");

        if n >= WARPS as usize {
            if n > MMA_TOKENS {
                return Err(EngineError::Unsupported(format!(
                    "prompt matmul holds {MMA_TOKENS} tokens, got {n}"
                )));
            }
            let owned;
            let prepared = match prepared {
                Some(prepared) => prepared,
                None => {
                    owned = self.prepare(x, x_offset, n, k)?;
                    &owned
                }
            };
            // A single matrix is a group of one, so the residual add fits in the
            // epilogue instead of costing a separate pass over the hidden state.
            return self.matmul_prompt_group(
                &[GroupEntry {
                    identity,
                    bytes: weights,
                    out,
                    y,
                    y_offset,
                }],
                prepared,
                n,
                k,
                bias,
            );
        }
        let (scales, quants) = self.quantized(identity, weights)?;
        let mut args = [
            Arg::Pointer(scales),
            Arg::Pointer(quants),
            Arg::Pointer(shifted(x.pointer, x_offset)),
            Arg::Pointer(shifted(y.pointer, y_offset)),
            Arg::Pointer(bias.map_or(shifted(y.pointer, y_offset), |(p, o)| shifted(p, o))),
            Arg::Int(bias.is_some() as i32),
            Arg::Int(k as i32),
            Arg::Int(out as i32),
            Arg::Int(n as i32),
        ];
        // Single-token decode gives every output row its own warp, four rows per
        // block: the row is small, so the win is in flight, not in width.
        if n == 1 {
            return self.matmul_decode_group(
                &[GroupEntry {
                    identity,
                    bytes: weights,
                    out,
                    y,
                    y_offset,
                }],
                x,
                x_offset,
                k,
                bias,
            );
        }
        // Two to seven tokens keep the eight-warp kernel; the bias is applied
        // in its epilogue like the other two kernels.
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

    /// Multiply an activation by `sigmoid(gate)`, elementwise.
    pub fn sigmoid_mul(
        &self,
        act: &DeviceBuffer,
        gate: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngineError> {
        let mut args = [
            Arg::Pointer(act.pointer),
            Arg::Pointer(gate.pointer),
            Arg::Int(n as i32),
        ];
        self.launch(
            self.kernels.sigmoid_mul,
            (blocks(n), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    /// Split a packed query projection into its query and gate halves.
    pub fn split_query_gate(
        &self,
        packed: &DeviceBuffer,
        query: &DeviceBuffer,
        gate: &DeviceBuffer,
        heads: usize,
        head_dim: usize,
        tokens: usize,
    ) -> Result<(), EngineError> {
        let total = tokens * heads * head_dim;
        let mut args = [
            Arg::Pointer(packed.pointer),
            Arg::Pointer(query.pointer),
            Arg::Pointer(gate.pointer),
            Arg::Int(heads as i32),
            Arg::Int(head_dim as i32),
            Arg::Int(total as i32),
        ];
        self.launch(
            self.kernels.split_query_gate,
            (blocks(total), 1, 1),
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
        self.axpby_f32(y, y_offset, x, x_offset, 1.0, 1.0, n)
    }

    /// Copies a contiguous f32 range on the device using four-wide loads.
    #[allow(dead_code)]
    pub fn copy_f32(
        &self,
        destination: &DeviceBuffer,
        destination_offset: usize,
        source: &DeviceBuffer,
        source_offset: usize,
        n: usize,
    ) -> Result<(), EngineError> {
        let mut args = [
            Arg::Pointer(shifted(destination.pointer, destination_offset)),
            Arg::Pointer(shifted(source.pointer, source_offset)),
            Arg::Int(n as i32),
        ];
        self.launch(
            self.kernels.copy_f32,
            (blocks(n), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    /// Fills a contiguous f32 range on the device.
    #[allow(dead_code)]
    pub fn fill_f32(
        &self,
        destination: &DeviceBuffer,
        destination_offset: usize,
        value: f32,
        n: usize,
    ) -> Result<(), EngineError> {
        let mut args = [
            Arg::Pointer(shifted(destination.pointer, destination_offset)),
            Arg::Float(value),
            Arg::Int(n as i32),
        ];
        self.launch(
            self.kernels.fill_f32,
            (blocks(n), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    /// Computes `y = alpha*y + beta*x` for a contiguous f32 range.
    #[allow(clippy::too_many_arguments)]
    pub fn axpby_f32(
        &self,
        y: &DeviceBuffer,
        y_offset: usize,
        x: &DeviceBuffer,
        x_offset: usize,
        alpha: f32,
        beta: f32,
        n: usize,
    ) -> Result<(), EngineError> {
        let mut args = [
            Arg::Pointer(shifted(y.pointer, y_offset)),
            Arg::Pointer(shifted(x.pointer, x_offset)),
            Arg::Float(alpha),
            Arg::Float(beta),
            Arg::Int(n as i32),
        ];
        self.launch(
            self.kernels.axpby_f32,
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
    /// Query rotation, key rotation straight into the cache, and the value copy,
    /// in one launch: the three passes the layer needs before attention.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_append(
        &self,
        query: &DeviceBuffer,
        key: &DeviceBuffer,
        value: &DeviceBuffer,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        base: f32,
        kv_dim: usize,
    ) -> Result<(), EngineError> {
        let total = tokens * q_heads * (head_dim / 2);
        let (position, _) = self.state_pointers()?;
        let mut args = [
            Arg::Pointer(query.pointer),
            Arg::Pointer(key.pointer),
            Arg::Pointer(value.pointer),
            Arg::Pointer(keys.pointer),
            Arg::Pointer(values.pointer),
            Arg::Int(tokens as i32),
            Arg::Int(q_heads as i32),
            Arg::Int(kv_heads as i32),
            Arg::Int(head_dim as i32),
            Arg::Int((head_dim / 2) as i32),
            Arg::Int(n_rot as i32),
            Arg::Float(base),
            Arg::Pointer(position),
            Arg::Int(kv_dim as i32),
        ];
        self.launch(
            self.kernels.rope_append,
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

    /// Prefill attention with the keys and values staged in shared memory and the
    /// softmax kept online, one block per head and sixteen query positions.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill(
        &self,
        query: &DeviceBuffer,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        out: &DeviceBuffer,
        geometry: AttentionGeometry,
        start: usize,
        tokens: usize,
    ) -> Result<(), EngineError> {
        let mut args = [
            Arg::Pointer(query.pointer),
            Arg::Pointer(keys.pointer),
            Arg::Pointer(values.pointer),
            Arg::Pointer(out.pointer),
            Arg::Int(geometry.n_head as i32),
            Arg::Int(geometry.n_head_kv as i32),
            Arg::Int(geometry.head_dim as i32),
            Arg::Int(start as i32),
            Arg::Int(tokens as i32),
            Arg::Float(geometry.scale),
        ];
        self.launch(
            self.kernels.attention_prefill,
            (
                geometry.n_head.div_ceil(PREFILL_HEADS) as u32,
                tokens.div_ceil(PREFILL_QUERY_TILE) as u32,
                1,
            ),
            (LANES, PREFILL_WARPS as u32, 1),
            PREFILL_SHARED_BYTES,
            &mut args,
        )
    }

    /// Prefill attention on the tensor cores: eight heads that share one KV
    /// head, sixteen queries, and the same shared-memory budget as the fp32
    /// kernel because queries are staged as f16.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_mma(
        &self,
        query: &DeviceBuffer,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        out: &DeviceBuffer,
        geometry: AttentionGeometry,
        start: usize,
        tokens: usize,
    ) -> Result<(), EngineError> {
        const MMA_HEADS: usize = 8;
        const MMA_QUERY_TILE: usize = 16;
        let mut args = [
            Arg::Pointer(query.pointer),
            Arg::Pointer(keys.pointer),
            Arg::Pointer(values.pointer),
            Arg::Pointer(out.pointer),
            Arg::Int(geometry.n_head as i32),
            Arg::Int(geometry.n_head_kv as i32),
            Arg::Int(geometry.head_dim as i32),
            Arg::Int(start as i32),
            Arg::Int(tokens as i32),
            Arg::Float(geometry.scale),
        ];
        self.launch(
            self.kernels.attention_prefill_mma,
            (
                geometry.n_head.div_ceil(MMA_HEADS) as u32,
                tokens.div_ceil(MMA_QUERY_TILE) as u32,
                1,
            ),
            (LANES, 8, 1),
            PREFILL_SHARED_BYTES,
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

    /// Decode attention: positions split across blocks, then combined.
    ///
    /// `visible` sets the split count and therefore the grid, so a captured step
    /// is only valid for the grid it was recorded with — the caller keys its
    /// graph cache on [`split_count`]. The partials buffer is sized for
    /// `capacity` instead, so its address does not move as the context grows.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_split(
        &self,
        query: &DeviceBuffer,
        keys: &DeviceBuffer,
        values: &DeviceBuffer,
        out: &DeviceBuffer,
        geometry: AttentionGeometry,
        visible: usize,
        capacity: usize,
    ) -> Result<(), EngineError> {
        let splits = split_count(visible);
        let (_, visible) = self.state_pointers()?;
        let heads = geometry.n_head;
        let widest = split_count(capacity);
        let floats = heads * widest * geometry.head_dim + heads * widest * 2;
        let workspace = self.workspace(floats)?;

        let mut args = [
            Arg::Pointer(query.pointer),
            Arg::Pointer(keys.pointer),
            Arg::Pointer(values.pointer),
            Arg::Pointer(workspace),
            Arg::Pointer(shifted(workspace, heads * splits * geometry.head_dim)),
            Arg::Pointer(shifted(
                workspace,
                heads * splits * geometry.head_dim + heads * splits,
            )),
            Arg::Int(geometry.n_head as i32),
            Arg::Int(geometry.n_head_kv as i32),
            Arg::Int(geometry.head_dim as i32),
            Arg::Pointer(visible),
            Arg::Int(splits as i32),
            Arg::Float(geometry.scale),
        ];
        self.launch(
            self.kernels.attention_split,
            (heads as u32, splits as u32, 1),
            (ATTENTION_THREADS, 1, 1),
            (MAX_SPLIT_POSITIONS * 4) as u32,
            &mut args,
        )?;

        let mut combine_args = [
            Arg::Pointer(workspace),
            Arg::Pointer(shifted(workspace, heads * splits * geometry.head_dim)),
            Arg::Pointer(shifted(
                workspace,
                heads * splits * geometry.head_dim + heads * splits,
            )),
            Arg::Pointer(out.pointer),
            Arg::Int(geometry.head_dim as i32),
            Arg::Int(splits as i32),
        ];
        self.launch(
            self.kernels.attention_combine,
            (heads as u32, 1, 1),
            (ATTENTION_THREADS, 1, 1),
            0,
            &mut combine_args,
        )
    }

    /// int8 against int8: activations are quantized once per matmul, then the
    /// prompt kernel multiplies them with the weights and applies the per-block
    /// scales.
    #[allow(clippy::too_many_arguments)]
    /// Quantises activations once, so several matmuls reading the same rows — the
    /// q, k and v projections, or the two feed-forward ones — pay for it once
    /// instead of once each.
    pub fn prepare(
        &self,
        x: &DeviceBuffer,
        x_offset: usize,
        n: usize,
        k: usize,
    ) -> Result<Prepared, EngineError> {
        let blocks = k / 32;
        let quants_bytes = n * k;
        let buffer = self.activation_workspace(quants_bytes, n * blocks)?;
        // One thread per thirty-two-element block, so the grid is the whole
        // activation and each thread streams its own hundred and twenty-eight
        // bytes contiguously.
        let total = (blocks * n) as u32;
        let mut quantize_args = [
            Arg::Pointer(shifted(x.pointer, x_offset)),
            Arg::Pointer(buffer),
            Arg::Pointer(shifted(buffer, quants_bytes / 4)),
            Arg::Int(k as i32),
            Arg::Int(total as i32),
        ];
        self.launch(
            self.kernels.quantize_rows,
            (total.div_ceil(128), 1, 1),
            (128, 1, 1),
            0,
            &mut quantize_args,
        )?;
        Ok(Prepared(buffer))
    }

    /// `quants = quantize(rmsnorm(src))` for the projections that read the
    /// normalised rows, so the f32 norm is never written out and read back.
    pub fn rmsnorm_prepare(
        &self,
        source: &DeviceBuffer,
        source_offset: usize,
        weight: &[f32],
        n: usize,
        rows: usize,
        eps: f32,
    ) -> Result<Prepared, EngineError> {
        let identity = (weight.as_ptr() as usize, weight.len());
        let device_weight = self.plain(identity, as_bytes(weight))?;
        let blocks = n / 32;
        let quants_bytes = rows * n;
        let buffer = self.activation_workspace(quants_bytes, rows * blocks)?;
        let mut args = [
            Arg::Pointer(shifted(source.pointer, source_offset)),
            Arg::Pointer(device_weight),
            Arg::Pointer(buffer),
            Arg::Pointer(shifted(buffer, quants_bytes / 4)),
            Arg::Int(n as i32),
            Arg::Int(rows as i32),
            Arg::Float(eps),
        ];
        self.launch(
            self.kernels.rmsnorm_quantize,
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )?;
        Ok(Prepared(buffer))
    }

    /// `quants = quantize(silu(gate) * up)` for the down projection, so the
    /// activation is never written out and read back in f32.
    pub fn silu_mul_quantize(
        &self,
        gate: &DeviceBuffer,
        up: &DeviceBuffer,
        n: usize,
        k: usize,
    ) -> Result<Prepared, EngineError> {
        let blocks = k / 32;
        let quants_bytes = n * k;
        let buffer = self.activation_workspace(quants_bytes, n * blocks)?;
        let total = (blocks * n) as u32;
        let mut quantize_args = [
            Arg::Pointer(gate.pointer),
            Arg::Pointer(up.pointer),
            Arg::Pointer(buffer),
            Arg::Pointer(shifted(buffer, quants_bytes / 4)),
            Arg::Int(k as i32),
            Arg::Int(total as i32),
        ];
        self.launch(
            self.kernels.silu_mul_quantize,
            (total.div_ceil(128), 1, 1),
            (128, 1, 1),
            0,
            &mut quantize_args,
        )?;
        Ok(Prepared(buffer))
    }

    /// One launch for up to four single-token matmuls that share the activation
    /// vector, their row spaces concatenated. `bias` is the residual a decode step
    /// adds in the epilogue; it belongs to a single matrix, so it is only set when
    /// the group holds one.
    pub fn matmul_decode_group(
        &self,
        group: &[GroupEntry<'_>],
        x: &DeviceBuffer,
        x_offset: usize,
        k: usize,
        bias: Option<(DevicePtr, usize)>,
    ) -> Result<(), EngineError> {
        assert!(
            !group.is_empty() && group.len() <= MMA_GROUP_MAX,
            "decode matmul group holds one to four matrices"
        );
        let mut resolved = Vec::with_capacity(group.len());
        let mut rows = 0;
        for entry in group {
            assert!(
                entry.out.is_multiple_of(DECODE_ROWS_PER_BLOCK),
                "decode matmul rows come in blocks of four"
            );
            let (scales, quants) = self.quantized(entry.identity, entry.bytes)?;
            rows += entry.out;
            resolved.push((
                scales,
                quants,
                entry.out,
                shifted(entry.y.pointer, entry.y_offset),
            ));
        }
        while resolved.len() < MMA_GROUP_MAX {
            resolved.push((resolved[0].0, resolved[0].1, 0, resolved[0].3));
        }
        let mut args = [
            Arg::Pointer(resolved[0].0),
            Arg::Pointer(resolved[0].1),
            Arg::Pointer(resolved[1].0),
            Arg::Pointer(resolved[1].1),
            Arg::Pointer(resolved[2].0),
            Arg::Pointer(resolved[2].1),
            Arg::Pointer(resolved[3].0),
            Arg::Pointer(resolved[3].1),
            Arg::Pointer(resolved[0].3),
            Arg::Pointer(resolved[1].3),
            Arg::Pointer(resolved[2].3),
            Arg::Pointer(resolved[3].3),
            Arg::Int(resolved[0].2 as i32),
            Arg::Int(resolved[1].2 as i32),
            Arg::Int(resolved[2].2 as i32),
            Arg::Int(resolved[3].2 as i32),
            Arg::Int(group.len() as i32),
            Arg::Pointer(shifted(x.pointer, x_offset)),
            Arg::Pointer(bias.map_or(shifted(resolved[0].3, 0), |(p, o)| shifted(p, o))),
            Arg::Int(bias.is_some() as i32),
            Arg::Int(k as i32),
        ];
        self.launch(
            self.kernels.matmul_decode,
            ((rows / DECODE_ROWS_PER_BLOCK) as u32, 1, 1),
            ((DECODE_ROWS_PER_BLOCK as u32) * LANES, 1, 1),
            0,
            &mut args,
        )
    }

    /// One launch for up to four matmuls that share the same activations. Their
    /// row tiles are concatenated, so the group fills a wave together instead of
    /// each matrix filling one on its own.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_prompt_group(
        &self,
        group: &[GroupEntry<'_>],
        prepared: &Prepared,
        n: usize,
        k: usize,
        bias: Option<(DevicePtr, usize)>,
    ) -> Result<(), EngineError> {
        if n > MMA_TOKENS {
            return Err(EngineError::Unsupported(format!(
                "prompt matmul holds {MMA_TOKENS} tokens, got {n}"
            )));
        }
        assert!(
            !group.is_empty() && group.len() <= MMA_GROUP_MAX,
            "prompt matmul group holds one to four matrices"
        );
        assert!(
            bias.is_none() || group.len() == 1,
            "a residual add belongs to a single-matrix group"
        );
        let act_quants = prepared.0;
        let act_scales = shifted(prepared.0, n * k / 4);
        let mut resolved = Vec::with_capacity(group.len());
        let mut row_tiles = 0;
        for entry in group {
            let (scales, quants) = self.quantized(entry.identity, entry.bytes)?;
            row_tiles += entry.out.div_ceil(MMA_ROWS * MMA_WARPS);
            resolved.push((
                scales,
                quants,
                entry.out,
                shifted(entry.y.pointer, entry.y_offset),
            ));
        }
        while resolved.len() < MMA_GROUP_MAX {
            resolved.push((resolved[0].0, resolved[0].1, 0, resolved[0].3));
        }
        let token_tiles = n.div_ceil(MMA_BLOCK_TOKENS);
        let plan = wave_plan(row_tiles, token_tiles, self.blocks_per_wave());
        let mut args = [
            Arg::Pointer(resolved[0].0),
            Arg::Pointer(resolved[0].1),
            Arg::Pointer(resolved[1].0),
            Arg::Pointer(resolved[1].1),
            Arg::Pointer(resolved[2].0),
            Arg::Pointer(resolved[2].1),
            Arg::Pointer(resolved[3].0),
            Arg::Pointer(resolved[3].1),
            Arg::Pointer(resolved[0].3),
            Arg::Pointer(resolved[1].3),
            Arg::Pointer(resolved[2].3),
            Arg::Pointer(resolved[3].3),
            Arg::Int(resolved[0].2 as i32),
            Arg::Int(resolved[1].2 as i32),
            Arg::Int(resolved[2].2 as i32),
            Arg::Int(resolved[3].2 as i32),
            Arg::Int(group.len() as i32),
            Arg::Pointer(act_scales),
            Arg::Pointer(act_quants),
            Arg::Int(k as i32),
            Arg::Int(n as i32),
            Arg::Int(token_tiles as i32),
            Arg::Int(plan.full_blocks as i32),
            Arg::Int(plan.tail_slices as i32),
            Arg::Pointer(bias.map_or(resolved[0].3, |(pointer, offset)| shifted(pointer, offset))),
            Arg::Int(bias.is_some() as i32),
        ];
        self.launch(
            self.kernels.matmul_mma,
            // One flat grid, token tiles the fast dimension so the blocks resident
            // together share a weight tile and hit L2 instead of re-reading it.
            (plan.blocks as u32, 1, 1),
            (LANES, MMA_WARPS as u32, 1),
            MMA_SHARED_BYTES,
            &mut args,
        )
    }

    /// `scale_floats` for the per-block scales, in one cached allocation.
    fn activation_workspace(
        &self,
        quants_bytes: usize,
        scale_floats: usize,
    ) -> Result<DevicePtr, EngineError> {
        let floats = quants_bytes.div_ceil(4) + scale_floats;
        let mut slot = self.activation.lock().expect("activation lock");
        if let Some(buffer) = slot.as_ref()
            && buffer.len() >= floats
        {
            return Ok(buffer.pointer);
        }
        let buffer = self.alloc(floats)?;
        let pointer = buffer.pointer;
        *slot = Some(buffer);
        Ok(pointer)
    }

    /// Scratch that survives between launches, so a per-layer call does not
    /// allocate. Grows to the largest request and is reused after that.
    fn workspace(&self, floats: usize) -> Result<DevicePtr, EngineError> {
        let mut slot = self.workspace.lock().expect("workspace lock");
        if let Some(buffer) = slot.as_ref()
            && buffer.len() >= floats
        {
            return Ok(buffer.pointer);
        }
        let buffer = self.alloc(floats)?;
        let pointer = buffer.pointer;
        *slot = Some(buffer);
        Ok(pointer)
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

    /// Whole prompt embedding in one launch: `tokens` is copied into a cached
    /// device buffer, then one grid dequantises every row it names.
    pub fn embeddings(
        &self,
        identity: (usize, usize),
        weights: &[u8],
        tokens: &[u32],
        out: &DeviceBuffer,
        out_offset: usize,
        k: usize,
    ) -> Result<(), EngineError> {
        let (scales, quants) = self.quantized(identity, weights)?;
        let token_buffer = self.token_buffer(tokens)?;
        let mut args = [
            Arg::Pointer(scales),
            Arg::Pointer(quants),
            Arg::Pointer(token_buffer),
            Arg::Pointer(shifted(out.pointer, out_offset)),
            Arg::Int(tokens.len() as i32),
            Arg::Int(k as i32),
        ];
        self.launch(
            self.kernels.embedding_rows,
            (blocks(tokens.len() * k), 1, 1),
            (256, 1, 1),
            0,
            &mut args,
        )
    }

    /// Cached device copy of the token ids, grown on demand.
    fn token_buffer(&self, tokens: &[u32]) -> Result<DevicePtr, EngineError> {
        self.bind()?;
        let mut slot = self.tokens.lock().expect("tokens lock");
        if let Some(buffer) = slot.as_ref()
            && buffer.len() >= tokens.len()
        {
            // SAFETY: both pointers are valid for the copied length.
            let status = self.copy_to_device(
                buffer.pointer,
                tokens.as_ptr().cast::<c_void>(),
                std::mem::size_of_val(tokens),
            );
            if status == SUCCESS {
                return Ok(buffer.pointer);
            }
            return Err(EngineError::Unsupported(format!(
                "cuda: token upload failed: {}",
                error_message(status)
            )));
        }
        let buffer = self.alloc(tokens.len().max(1))?;
        // SAFETY: both pointers are valid for the copied length.
        let status = self.copy_to_device(
            buffer.pointer,
            tokens.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(tokens),
        );
        if status != SUCCESS {
            return Err(EngineError::Unsupported(format!(
                "cuda: token upload failed: {}",
                error_message(status)
            )));
        }
        let pointer = buffer.pointer;
        *slot = Some(buffer);
        Ok(pointer)
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
        let device_scales = self.upload_new_bytes(as_bytes(&scales))?;
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

    fn upload_new_bytes(&self, bytes: &[u8]) -> Result<DeviceBuffer, EngineError> {
        self.bind()?;
        let buffer = self.alloc(bytes.len().div_ceil(4))?;
        // SAFETY: the buffer holds at least `bytes.len()` bytes.
        let status =
            self.copy_to_device(buffer.pointer, bytes.as_ptr().cast::<c_void>(), bytes.len());
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
        let kernel = self
            .names
            .iter()
            .find(|(handle, _)| handle.0 == function.0)
            .map_or("unknown", |(_, name)| *name);
        // The launch and the kernel signature are two places that have to agree,
        // and nothing in the driver checks it: `cuLaunchKernel` reads as many
        // parameters as the kernel declares, whatever the caller passed. Checked
        // where it costs nothing, which is every test run.
        if cfg!(debug_assertions) {
            let declared = self
                .params
                .iter()
                .find(|(handle, _)| handle.0 == function.0)
                .map_or(-1, |(_, count)| *count);
            debug_assert!(
                declared == args.len() as i32,
                "{kernel} declares {declared} parameters, the launch passes {}",
                args.len()
            );
        }
        trace!(
            kernel,
            grid = ?grid,
            block = ?block,
            shared_bytes,
            args = args.len(),
            "launching CUDA kernel"
        );
        let start = if self.profile_enabled {
            let event = self.event()?;
            // SAFETY: the event was just created and the stream is the default.
            unsafe {
                cuEventRecord(event, self.stream);
            }
            Some(event)
        } else {
            None
        };
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
                self.stream,
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
        debug!(kernel, "CUDA kernel queued");
        if let Some(start) = start {
            self.record(start, function);
        }
        Ok(())
    }

    fn event(&self) -> Result<Event, EngineError> {
        let mut event: Event = std::ptr::null_mut();
        // SAFETY: the driver writes one handle into a pointer we own.
        let status = unsafe { cuEventCreate(&mut event, 0) };
        if status != 0 {
            return Err(EngineError::Unsupported(format!(
                "cuda: cuEventCreate failed with status {status}"
            )));
        }
        Ok(event)
    }

    /// Closes the window opened before the launch and adds it to the table.
    fn record(&self, start: Event, function: Function) {
        let Ok(end) = self.event() else {
            return;
        };
        // SAFETY: both events exist and the stream is the default one.
        unsafe {
            cuEventRecord(end, self.stream);
            cuEventSynchronize(end);
        }
        let mut milliseconds = 0.0f32;
        // SAFETY: two recorded events and one f32 out-parameter.
        let status = unsafe { cuEventElapsedTime(&mut milliseconds, start, end) };
        let name = self
            .names
            .iter()
            .find(|(handle, _)| handle.0 == function.0)
            .map_or("unknown", |(_, name)| *name);
        if status == 0
            && let Some(table) = self.profile.lock().expect("profile lock").as_mut()
        {
            trace!(
                kernel = name,
                elapsed_ms = milliseconds,
                "CUDA kernel complete"
            );
            let slot = table.entry(name).or_insert((0.0, 0));
            slot.0 += f64::from(milliseconds);
            slot.1 += 1;
        }
        // SAFETY: both events were created here and are no longer used.
        unsafe {
            cuEventDestroy(start);
            cuEventDestroy(end);
        }
    }

    /// Prints the table gathered under `LLM_PROFILE` and clears it.
    pub fn report_profile(&self) {
        let Some(table) = self.profile.lock().expect("profile lock").take() else {
            return;
        };
        let mut rows: Vec<(&str, f64, u64)> = table
            .iter()
            .map(|(name, (ms, calls))| (*name, *ms, *calls))
            .collect();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f64 = rows.iter().map(|(_, ms, _)| *ms).sum();
        eprintln!("kernel              total ms   calls   ms/call");
        for (name, ms, calls) in &rows {
            eprintln!(
                "{name:<18} {ms:9.2} {calls:7} {:9.4}",
                ms / (*calls).max(1) as f64
            );
        }
        eprintln!("{:<18} {total:9.2}", "sum");
    }
}

// SAFETY: every driver call binds the context to the calling thread first, and
// the engine runs one generation at a time, so the handles are never used
// concurrently. The kernel launches are serialised by the driver's own stream.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.report_profile();
        self.weights.lock().expect("weight cache lock").clear();
        self.state.lock().expect("state lock").take();
        // SAFETY: the stream and context were created by this struct and are
        // destroyed once, stream first.
        unsafe {
            let _ = cuStreamSynchronize(self.stream);
            let _ = cuStreamDestroy_v2(self.stream);
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

/// How a prompt matmul fills the device.
///
/// A tile is a whole unit of work: 128 output rows by up to 128 tokens. When the
/// tile count is not a multiple of the blocks that run at once, the last wave is
/// nearly empty and still costs a full tile, which at 512 tokens is 20% of the
/// matmul. The tail tiles are therefore cut along the token axis, each slice
/// writing disjoint columns, so no partial sums need combining.
struct WavePlan {
    blocks: usize,
    full_blocks: usize,
    tail_slices: usize,
}

fn wave_plan(row_tiles: usize, token_tiles: usize, multiprocessors: usize) -> WavePlan {
    let tiles = row_tiles * token_tiles;
    let waves = tiles.div_ceil(multiprocessors);
    let full_blocks = waves.saturating_sub(1) * multiprocessors;
    let tail = tiles - full_blocks;
    // Slices must divide a tile evenly and stay a multiple of the mma token step.
    let mut tail_slices = 1;
    if tail < multiprocessors {
        // Cut each tail tile into as many pieces as the last wave has room for,
        // rounded down to a power of two so the pieces stay a multiple of the
        // mma's eight tokens.
        let room = multiprocessors / tail.max(1);
        tail_slices = (1usize << (usize::BITS - 1 - room.leading_zeros())).clamp(1, 8);
    }
    WavePlan {
        blocks: full_blocks + tail * tail_slices,
        full_blocks,
        tail_slices,
    }
}

/// Shape of the attention kernel's grid, kept in one place.
pub struct AttentionGeometry {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub scores_stride: usize,
    pub scale: f32,
}

/// Raises the dynamic shared memory one kernel may ask for, past the 48 KiB a
/// kernel gets without it.
fn reserve_shared(function: Function, bytes: u32) -> Result<Function, EngineError> {
    // SAFETY: `function` came from cuModuleGetFunction; the driver validates the
    // value against the device limit and reports failure through the status.
    let status =
        unsafe { cuFuncSetAttribute(function, FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED, bytes as i32) };
    if status != 0 {
        return Err(EngineError::Unsupported(format!(
            "cuda: raising shared memory to {bytes} failed with status {status}"
        )));
    }
    Ok(function)
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
/// The scales stay f16 — that is their precision in the file, and it halves the
/// scale traffic on every weight pass.
fn repack_q8(bytes: &[u8]) -> (Vec<f16>, Vec<i8>) {
    const BLOCK: usize = 34;
    let blocks = bytes.len() / BLOCK;
    let mut scales = Vec::with_capacity(blocks);
    let mut quants = Vec::with_capacity(blocks * 32);
    for block in bytes.as_chunks::<BLOCK>().0 {
        let bits = u16::from_le_bytes([block[0], block[1]]);
        scales.push(f16::from_bits(bits));
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
