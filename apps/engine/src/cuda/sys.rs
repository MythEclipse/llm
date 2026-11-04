//! Raw CUDA driver API bindings.
//!
//! Only the calls the engine needs: one context, flat device buffers, a PTX
//! module, and synchronous kernel launches. Linking against `libcuda.so` keeps
//! the surface small; no CUDA runtime library is involved.

use std::ffi::{c_char, c_void};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device(pub i32);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Context(pub *mut c_void);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Module(pub *mut c_void);

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Function(pub *mut c_void);

/// `CUevent` is an opaque driver pointer.
pub type Event = *mut c_void;

/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` from the driver API.
pub const FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED: i32 = 8;

/// A device pointer, passed to kernels by value.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePtr(pub u64);

pub const SUCCESS: i32 = 0;

/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`: how many blocks can run at once.
pub const ATTRIBUTE_MULTIPROCESSORS: i32 = 16;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: u32) -> i32;
    pub fn cuDeviceGet(device: *mut Device, ordinal: i32) -> i32;
    pub fn cuDeviceGetName(name: *mut c_char, len: i32, device: Device) -> i32;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, device: Device) -> i32;
    /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT` is 16 in the driver API.
    pub fn cuDeviceGetAttribute(value: *mut i32, attribute: i32, device: Device) -> i32;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> i32;
    pub fn cuCtxCreate_v2(context: *mut Context, flags: u32, device: Device) -> i32;
    pub fn cuCtxDestroy_v2(context: Context) -> i32;
    pub fn cuCtxSetCurrent(context: Context) -> i32;
    pub fn cuMemAlloc_v2(pointer: *mut DevicePtr, bytes: usize) -> i32;
    pub fn cuMemFree_v2(pointer: DevicePtr) -> i32;
    pub fn cuMemcpyHtoD_v2(destination: DevicePtr, source: *const c_void, bytes: usize) -> i32;
    pub fn cuMemcpyDtoH_v2(destination: *mut c_void, source: DevicePtr, bytes: usize) -> i32;
    pub fn cuModuleLoadData(module: *mut Module, image: *const c_void) -> i32;
    pub fn cuModuleGetFunction(function: *mut Function, module: Module, name: *const c_char)
    -> i32;
    /// Raises the dynamic shared memory a kernel may ask for. Without it a launch
    /// over 48 KiB fails, which is where the tiled kernels land.
    pub fn cuFuncSetAttribute(function: Function, attribute: i32, value: i32) -> i32;
    /// Device timestamps, so a kernel can be timed where it runs instead of where
    /// it is queued.
    pub fn cuEventCreate(event: *mut Event, flags: u32) -> i32;
    pub fn cuEventRecord(event: Event, stream: *mut c_void) -> i32;
    pub fn cuEventSynchronize(event: Event) -> i32;
    pub fn cuEventElapsedTime(milliseconds: *mut f32, start: Event, end: Event) -> i32;
    pub fn cuEventDestroy(event: Event) -> i32;
    pub fn cuLaunchKernel(
        function: Function,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_bytes: u32,
        stream: *mut c_void,
        arguments: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    pub fn cuGetErrorString(error: i32, message: *mut *const c_char) -> i32;
}

/// Human-readable name for a driver error code.
pub fn error_message(code: i32) -> String {
    let mut message: *const c_char = std::ptr::null();
    // SAFETY: the driver only writes a pointer to a static string.
    let status = unsafe { cuGetErrorString(code, &raw mut message) };
    if status != SUCCESS || message.is_null() {
        return format!("CUDA error {code}");
    }
    // SAFETY: the driver returns a NUL-terminated static string.
    unsafe { std::ffi::CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}
