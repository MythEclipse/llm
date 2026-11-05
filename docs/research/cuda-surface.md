# Permukaan CUDA yang tersedia untuk engine ini

Inventaris apa saja yang bisa dipanggil dari kernel dan dari host di mesin ini,
disusun dari toolchain yang benar-benar terpasang, bukan dari ingatan: header
`/opt/cuda/include` (CUDA 13.4, `V13.4.92`), simbol `libcuda.so` (driver
615.78.08), dan `ptxas` untuk memutuskan instruksi mana yang sah untuk
`sm_89`. Lampiran di bawah dihasilkan dengan memindai ketiganya.

Kesimpulan yang menentukan pekerjaan kernel di repo ini:

- **Yang membatasi bukan header, tapi target.** Semua intrinsik CUDA yang
  relevan tersedia; yang tidak tersedia adalah instruksi `sm_90+`
  (`stmatrix`, `cp.async.bulk`/TMA, `elect.sync`, `wgmma`, `tcgen05`, cluster).
  Semuanya gagal di `ptxas` dengan pesan yang jelas, jadi batasnya bisa diuji,
  bukan diingat.
- **Host hanya punya driver API.** Repo ini tidak memakai CUDA Runtime
  (`cudaMalloc` dst.) dan tidak punya NVRTC. Dari 555 entry point driver, 32
  dipakai. Sisanya bukan kekurangan otomatis: banyak yang memang tidak relevan
  (texture, surface, interop grafis, green context, checkpoint).
- **Build hanya menghasilkan PTX.** `build.rs` menjalankan `nvcc -ptx`, dan
  driver JIT meng-compile-nya saat load. Artinya SASS tidak bisa diperiksa di
  build (perlu `-cubin`), dan register/occupancy baru terlihat di runtime.

## 1. Batas build ini

| Hal | Nilai di repo | Konsekuensi |
|---|---|---|
| Toolchain | `nvcc` 13.4, `-arch sm_89`, `-O3`, `-ptx` | instruksi divalidasi per target; SASS baru ada saat JIT |
| Runtime | `libcuda.so` (driver API) saja | 555 fungsi host, tanpa `cudaXxx` runtime |
| Target | `sm_89` (RTX 4050 Laptop, 20 SM) | tensor core `mma.sync` gen 4, tanpa TMA/cluster |
| Device code | C++ penuh lewat `nvcc` | intrinsik + inline PTX + template semuanya boleh |
| JIT | driver 615.78 | PTX 9.x diterima; `-cubin` tidak diperlukan |

Menambah satu panggilan driver berarti menambah deklarasi di
`apps/engine/src/cuda/sys.rs` lalu memakainya di `cuda/mod.rs`. Menambah satu
instruksi berarti menulis inline `asm volatile` di `kernels/gpu.cu` — pola yang
sudah dipakai untuk `mma.sync`, `ldmatrix`, dan `cp.async`.

## 2. Gerbang arsitektur, diuji dengan `ptxas`

Setiap baris di bawah adalah probe kecil yang dikompilasi ke `cubin` untuk
target yang disebut. Ini hasilnya di mesin ini, bukan kutipan dokumentasi:

| Instruksi / intrinsik | `sm_75` | `sm_80` | `sm_89` | `sm_90` |
|---|---|---|---|---|
| `cp.async` (`__pipeline_memcpy_async`) | — | OK | OK | OK |
| `cp.async.mbarrier.arrive` | gagal | OK | OK | OK |
| `ldmatrix.sync.aligned.m8n8` | OK | OK | OK | OK |
| `mma.sync.m16n8k16.f16` | OK | OK | OK | OK |
| `mma.sync.m16n8k8.tf32` | gagal | OK | OK | OK |
| `mma.sync.m16n8k32.s8` | gagal | OK | OK | OK |
| `__dp4a` (`dp4a.s32.s32`) | OK | OK | OK | OK |
| `mbarrier.init/arrive` | gagal | OK | OK | OK |
| `discard.global.L2` | — | OK | OK | OK |
| `prefetch.global.L2` | OK | OK | OK | OK |
| `__ldcs` (streaming load) | OK | OK | OK | OK |
| `__reduce_add_sync` | — | OK | OK | OK |
| `stmatrix.sync.aligned.m8n8` | gagal | gagal | **gagal** | OK |
| `cp.async.bulk` (TMA) | gagal | gagal | **gagal** | OK |
| `elect.sync` | gagal | gagal | **gagal** | OK |

Pesan `ptxas` untuk tiga yang gagal di `sm_89`:
`Feature 'stmatrix' requires .target sm_90 or higher`,
`Feature 'cp.async.bulk' requires .target sm_90 or higher`,
`Feature 'elect' requires .target sm_90 or higher`.

Yang tidak masuk tabel karena butuh target `a`/arsitektur lebih baru dan tidak
bisa dipakai di sini sama sekali: `wgmma` (`sm_90a`), `tcgen05.*` (`sm_100a`),
`setmaxnreg` (`sm_90a`), `multimem.*` (`sm_90a`), `cluster.*` dan
`barrier.cluster` (`sm_90`), `fabric.*` (`sm_100`).

## 3. Yang sudah dipakai repo ini

Kernel: 21 entry point di `apps/engine/kernels/gpu.cu`. Intrinsik yang muncul:

| Keluarga | Yang dipakai |
|---|---|
| warp | `__shfl_down_sync`, `__shfl_xor_sync`, `__shfl_sync` |
| sinkronisasi | `__syncthreads` |
| baca memori | `__ldg` |
| f16 | `__half`, `__half2`, `__float2half`, `__half2float`, `__floats2half2_rn`, `__half22float2` |
| konversi | `__float2int_rn` |
| inline PTX | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, `ldmatrix` (x1/x2/x2.trans), `cp.async` (satu tempat) |
| kualifier | `__global__`, `__device__`, `__shared__`, `__restrict__`, `__launch_bounds__`, `__forceinline__`, `__align__` |

Host: 32 dari 555 entry point driver — `cuInit`, `cuDevice*` (4),
`cuCtx*` (3), `cuMemAlloc/Free/GetInfo`, `cuMemcpy{HtoD,DtoH}Async`,
`cuModuleLoadData`, `cuModuleGetFunction`, `cuFuncSetAttribute`,
`cuLaunchKernel`, `cuStream*` (5), `cuEvent*` (5), `cuGraph*` (4),
`cuGetErrorString`.

## 4. Yang tersedia dan belum dipakai, diurutkan menurut nilainya di sini

### Device

| Yang tersedia | Kenapa relevan | Biaya / risiko |
|---|---|---|
| `cp.async` berpasangan `commit_group`/`wait_group` | matmul prompt masih menunggu tile di depan store pertama; sekarang baru satu tempat memakai `cp.async` | sedang: mengubah urutan staging bisa menurunkan atau menaikkan 1–2% — harus diukur, bukan diasumsikan |
| `ldmatrix.x4` | attention dan matmul memuat fragmen dengan x1/x2; x4 memotong jumlah instruksi LDS | rendah: bentuk fragmen harus cocok, mudah salah (bug transpos sudah pernah terjadi) |
| `__ldcs` / `__ldlu` untuk KV cache | KV dibaca sekali lalu tidak dipakai lagi; `evict-first` menghentikan ia mengusir bobot dari L2 | rendah, dan bisa diukur langsung dengan `bench` |
| `discard.global.L2` untuk buffer scratch | scratch ditimpa penuh tiap langkah; membuang isinya menghindari write-allocate | rendah; hanya `sm_80+`, sudah tersedia |
| `__reduce_add_sync` / `__reduce_max_sync` | pengurangan antar-lane manual dengan `__shfl_*` di beberapa kernel | rendah: hasilnya sama, satu instruksi |
| `mbarrier` + `cp.async.mbarrier.arrive` | overlap lebih halus daripada `__syncthreads` penuh | tinggi: menambah state per blok, mudah deadlock |
| `prefetch.global.L2` | bisa menyiapkan tile bobot layer berikutnya | sedang, dan L2 di kartu ini kecil (bobot 2,7 GB) |
| `__nv_associate_access_property` / `createpolicy` | kebijakan L2 per rentang (streaming untuk KV) | sedang; alternatif `__ldcs` lebih sederhana |
| `clock64()` / `__prof_trigger` | timing di dalam kernel tanpa event | rendah, hanya untuk diagnosis |
| `atomicAdd` f16/bf16 (`sm_70+`), `red.*` | belum perlu: tidak ada reduksi lintas blok di jalur ini | — |
| `wmma` (`nvcuda::wmma`) | pembungkus C++ dari `mma.sync` yang sudah ditulis tangan | tidak perlu: lapisan di atas instruksi yang sama |

### Host (driver)

| Entry point | Kenapa relevan | Biaya |
|---|---|---|
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | ukuran gelombang sekarang ditebak dari jumlah SM; ini menghitungnya dari register + shared per kernel | rendah, satu panggilan saat load |
| `cuFuncSetCacheConfig` | membagi L1/shared; kernel attention butuh shared besar, matmul butuh L1 | rendah |
| `cuStreamWriteValue32` | menulis posisi ke device langsung dari stream, menggantikan salinan 8 byte per token | rendah, menghapus satu operasi per token |
| `cuGraphAddKernelNode` / `cuGraphExecUpdate` | menyusun graph tanpa capture, dan memperbaruinya saat grid berubah — menghapus kebutuhan cache graph per bentuk | sedang; capture sekarang sudah bekerja, jadi ini hanya jika cache-nya jadi masalah |
| `cuMemPoolCreate` / `cuMemAllocAsync` | arena sekarang satu `cuMemAlloc` besar; pool berguna kalau nanti ada alokasi per permintaan | rendah tapi belum ada kebutuhan |
| `cuLaunchKernelEx` + launch attribute | prioritas stream, `programmatic dependent launch` (`sm_90+`), cluster (`sm_90+`) | rendah untuk prioritas, sisanya tidak tersedia |
| `cuMemPrefetchAsync` / `cuMemAdvise` | hanya untuk managed memory, yang tidak dipakai | — |
| `cuMemcpyBatchAsync` | menggabungkan banyak salinan kecil; sekarang salinan per token hanya 8 byte | rendah |
| `cuEventRecordWithFlags` (blocking sync) | spin-wait untuk latensi lebih rendah daripada `cuEventSynchronize` | rendah, hanya untuk pengukuran |
| `cuTensorMapEncodeTiled` (TMA) | butuh `sm_90+` | tidak tersedia |
| `cuLaunchCooperativeKernel` | sinkronisasi lintas blok; belum ada kernel yang butuh | — |
| VMM (`cuMemCreate`/`cuMemMap`), IPC, texture, surface, interop GL/EGL, green context, checkpoint/coredump | di luar kebutuhan satu proses satu GPU | — |

## 5. Cara memeriksa sendiri

Gerbang arsitektur di tabel bagian 2 dihasilkan dengan pola ini; pakai lagi
sebelum menulis kernel baru:

```bash
# satu probe = satu instruksi, tanpa penjaga __CUDA_ARCH__
cat > probe.cu <<'EOF'
extern "C" __global__ void p(float* out) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
               :: "r"(0u), "r"(0u), "r"(0u));
  out[threadIdx.x] = 0;
}
EOF
nvcc -arch=sm_89 -cubin -o /dev/null probe.cu   # gagal = tidak tersedia
```

`-cubin` penting: `-ptx` berhenti sebelum `ptxas`, jadi instruksi yang tidak sah
untuk target tetap lolos dan baru meledak saat JIT.

Untuk host, daftar lengkap entry point driver yang bisa dipanggil:

```bash
nm -D --defined-only /usr/lib/libcuda.so.1 | awk '{print $3}' | grep '^cu' | sort -u
```


## Lampiran A — 555 entry point driver API
Nama sudah dinormalkan: sufiks `_v2`/`_v3`/`_ptsz`/`_ptds` dilebur, karena
itu varian ABI dari fungsi yang sama. Yang bertanda **tebal** sudah dipakai
repo ini.

### init / device / context (66)

cuCtxAttach, **cuCtxCreate**, **cuCtxDestroy**, cuCtxDetach, cuCtxDisablePeerAccess, cuCtxEnablePeerAccess, cuCtxFromGreenCtx, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetCurrent, cuCtxGetDevResource, cuCtxGetDevice, cuCtxGetExecAffinity, cuCtxGetFlags, cuCtxGetId, cuCtxGetLimit, cuCtxGetSharedMemConfig, cuCtxGetStreamPriorityRange, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxRecordEvent, cuCtxResetPersistingL2Cache, cuCtxSetCacheConfig, **cuCtxSetCurrent**, cuCtxSetFlags, cuCtxSetLimit, cuCtxSetSharedMemConfig, cuCtxSynchronize, cuCtxWaitEvent, cuDeviceCanAccessPeer, cuDeviceComputeCapability, **cuDeviceGet**, **cuDeviceGetAttribute**, cuDeviceGetByPCIBusId, cuDeviceGetCliqueCount, cuDeviceGetCliqueInfo, cuDeviceGetCount, cuDeviceGetDefaultMemPool, cuDeviceGetDevResource, cuDeviceGetExecAffinitySupport, cuDeviceGetFabricClusterUuid, cuDeviceGetGraphMemAttribute, cuDeviceGetHostAtomicCapabilities, cuDeviceGetLuid, cuDeviceGetMemPool, **cuDeviceGetName**, cuDeviceGetNvSciSyncAttributes, cuDeviceGetP2PAtomicCapabilities, cuDeviceGetP2PAttribute, cuDeviceGetPCIBusId, cuDeviceGetProperties, cuDeviceGetTexture1DLinearMaxWidth, cuDeviceGetUuid, cuDeviceGraphMemTrim, cuDevicePrimaryCtxGetState, cuDevicePrimaryCtxRelease, cuDevicePrimaryCtxReset, cuDevicePrimaryCtxRetain, cuDevicePrimaryCtxSetFlags, cuDeviceRegisterAsyncNotification, cuDeviceSetGraphMemAttribute, cuDeviceSetMemPool, **cuDeviceTotalMem**, cuDeviceUnregisterAsyncNotification, cuDriverGetVersion, **cuInit**

### module, function, library, link (49)

cuFuncGetAttribute, cuFuncGetModule, cuFuncGetName, cuFuncGetParamCount, cuFuncGetParamInfo, cuFuncIsLoaded, cuFuncLoad, **cuFuncSetAttribute**, cuFuncSetBlockShape, cuFuncSetCacheConfig, cuFuncSetSharedMemConfig, cuFuncSetSharedSize, cuGetExportTable, cuGetProcAddress, cuKernelGetAttribute, cuKernelGetFunction, cuKernelGetLibrary, cuKernelGetName, cuKernelGetParamCount, cuKernelGetParamInfo, cuKernelSetAttribute, cuKernelSetCacheConfig, cuLibraryEnumerateKernels, cuLibraryGetGlobal, cuLibraryGetKernel, cuLibraryGetKernelCount, cuLibraryGetManaged, cuLibraryGetModule, cuLibraryGetUnifiedFunction, cuLibraryLoadData, cuLibraryLoadFromFile, cuLibraryUnload, cuLinkAddData, cuLinkAddFile, cuLinkComplete, cuLinkCreate, cuLinkDestroy, cuModuleEnumerateFunctions, **cuModuleGetFunction**, cuModuleGetFunctionCount, cuModuleGetGlobal, cuModuleGetLoadingMode, cuModuleGetSurfRef, cuModuleGetTexRef, cuModuleLoad, **cuModuleLoadData**, cuModuleLoadDataEx, cuModuleLoadFatBinary, cuModuleUnload

### memori datar (79)

cuMemAddressFree, cuMemAddressReserve, **cuMemAlloc**, cuMemAllocAsync, cuMemAllocFromPoolAsync, cuMemAllocHost, cuMemAllocManaged, cuMemAllocPitch, cuMemBatchDecompressAsync, cuMemCreate, **cuMemFree**, cuMemFreeAsync, cuMemFreeHost, cuMemGetAllocationGranularity, cuMemGetAllocationPropertiesFromHandle, cuMemGetHandleForAddressRange, **cuMemGetInfo**, cuMemGetMemPool, cuMemHostAlloc, cuMemHostGetDevicePointer, cuMemHostGetFlags, cuMemHostRegister, cuMemHostUnregister, cuMemMap, cuMemMapArrayAsync, cuMemPoolCreate, cuMemPoolDestroy, cuMemPoolExportPointer, cuMemPoolExportToShareableHandle, cuMemPoolGetAccess, cuMemPoolGetAttribute, cuMemPoolImportFromShareableHandle, cuMemPoolImportPointer, cuMemPoolSetAccess, cuMemPoolSetAttribute, cuMemPoolTrimTo, cuMemRelease, cuMemSetAccess, cuMemUnmap, cuMemcpy, cuMemcpy2D, cuMemcpy2DAsync, cuMemcpy2DUnaligned, cuMemcpy3D, cuMemcpy3DAsync, cuMemcpy3DBatchAsync, cuMemcpy3DPeer, cuMemcpy3DPeerAsync, cuMemcpy3DWithAttributesAsync, cuMemcpyAsync, cuMemcpyAtoA, cuMemcpyAtoD, cuMemcpyAtoH, cuMemcpyAtoHAsync, cuMemcpyBatchAsync, cuMemcpyDtoA, cuMemcpyDtoD, cuMemcpyDtoDAsync, cuMemcpyDtoH, **cuMemcpyDtoHAsync**, cuMemcpyHtoA, cuMemcpyHtoAAsync, cuMemcpyHtoD, **cuMemcpyHtoDAsync**, cuMemcpyPeer, cuMemcpyPeerAsync, cuMemcpyWithAttributesAsync, cuMemsetD16, cuMemsetD16Async, cuMemsetD2D16, cuMemsetD2D16Async, cuMemsetD2D32, cuMemsetD2D32Async, cuMemsetD2D8, cuMemsetD2D8Async, cuMemsetD32, cuMemsetD32Async, cuMemsetD8, cuMemsetD8Async

### managed, advice, prefetch (8)

cuMemAdvise, cuMemDiscardAndPrefetchBatchAsync, cuMemDiscardBatchAsync, cuMemPrefetchAsync, cuMemPrefetchBatchAsync, cuMemRangeGetAttribute, cuMemRangeGetAttributes, cuMemRetainAllocationHandle

### stream (32)

cuStreamAddCallback, cuStreamAttachMemAsync, cuStreamBatchMemOp, **cuStreamBeginCapture**, cuStreamBeginCaptureToCig, cuStreamBeginCaptureToGraph, cuStreamBeginRecaptureToGraph, cuStreamCopyAttributes, **cuStreamCreate**, cuStreamCreateWithPriority, **cuStreamDestroy**, **cuStreamEndCapture**, cuStreamEndCaptureToCig, cuStreamGetAttribute, cuStreamGetCaptureInfo, cuStreamGetCtx, cuStreamGetDevResource, cuStreamGetDevice, cuStreamGetFlags, cuStreamGetGreenCtx, cuStreamGetId, cuStreamGetPriority, cuStreamIsCapturing, cuStreamQuery, cuStreamSetAttribute, **cuStreamSynchronize**, cuStreamUpdateCaptureDependencies, cuStreamWaitEvent, cuStreamWaitValue32, cuStreamWaitValue64, cuStreamWriteValue32, cuStreamWriteValue64

### event (10)

cuDestroyExternalSemaphore, **cuEventCreate**, **cuEventDestroy**, **cuEventElapsedTime**, cuEventQuery, **cuEventRecord**, cuEventRecordWithFlags, **cuEventSynchronize**, cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync

### launch, occupancy, cache config (20)

cuLaunch, cuLaunchCooperativeKernel, cuLaunchCooperativeKernelMultiDevice, cuLaunchGrid, cuLaunchGridAsync, cuLaunchHostFunc, **cuLaunchKernel**, cuLaunchKernelEx, cuOccupancyAvailableDynamicSMemPerBlock, cuOccupancyMaxActiveBlocksPerMultiprocessor, cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, cuOccupancyMaxActiveClusters, cuOccupancyMaxPotentialBlockSize, cuOccupancyMaxPotentialBlockSizeWithFlags, cuOccupancyMaxPotentialClusterSize, cuParamSetSize, cuParamSetTexRef, cuParamSetf, cuParamSeti, cuParamSetv

### graph (96)

cuGraphAddBatchMemOpNode, cuGraphAddChildGraphNode, cuGraphAddDependencies, cuGraphAddEmptyNode, cuGraphAddEventRecordNode, cuGraphAddEventWaitNode, cuGraphAddExternalSemaphoresSignalNode, cuGraphAddExternalSemaphoresWaitNode, cuGraphAddHostNode, cuGraphAddKernelNode, cuGraphAddMemAllocNode, cuGraphAddMemFreeNode, cuGraphAddMemcpyNode, cuGraphAddMemsetNode, cuGraphAddNode, cuGraphBatchMemOpNodeGetParams, cuGraphBatchMemOpNodeSetParams, cuGraphChildGraphNodeGetGraph, cuGraphClone, cuGraphConditionalHandleCreate, cuGraphCreate, cuGraphDebugDotPrint, **cuGraphDestroy**, cuGraphDestroyNode, cuGraphEventRecordNodeGetEvent, cuGraphEventRecordNodeSetEvent, cuGraphEventWaitNodeGetEvent, cuGraphEventWaitNodeSetEvent, cuGraphExecBatchMemOpNodeSetParams, cuGraphExecChildGraphNodeSetParams, **cuGraphExecDestroy**, cuGraphExecEventRecordNodeSetEvent, cuGraphExecEventWaitNodeSetEvent, cuGraphExecExternalSemaphoresSignalNodeSetParams, cuGraphExecExternalSemaphoresWaitNodeSetParams, cuGraphExecGetFlags, cuGraphExecGetId, cuGraphExecHostNodeSetParams, cuGraphExecKernelNodeSetParams, cuGraphExecMemcpyNodeSetParams, cuGraphExecMemsetNodeSetParams, cuGraphExecNodeSetParams, cuGraphExecUpdate, cuGraphExternalSemaphoresSignalNodeGetParams, cuGraphExternalSemaphoresSignalNodeSetParams, cuGraphExternalSemaphoresWaitNodeGetParams, cuGraphExternalSemaphoresWaitNodeSetParams, cuGraphGetEdges, cuGraphGetId, cuGraphGetNodes, cuGraphGetRootNodes, cuGraphHostNodeGetParams, cuGraphHostNodeSetParams, **cuGraphInstantiate**, cuGraphInstantiateWithFlags, cuGraphInstantiateWithParams, cuGraphKernelNodeCopyAttributes, cuGraphKernelNodeGetAttribute, cuGraphKernelNodeGetParams, cuGraphKernelNodeSetAttribute, cuGraphKernelNodeSetParams, **cuGraphLaunch**, cuGraphMemAllocNodeGetParams, cuGraphMemFreeNodeGetParams, cuGraphMemcpyNodeGetParams, cuGraphMemcpyNodeSetParams, cuGraphMemsetNodeGetParams, cuGraphMemsetNodeSetParams, cuGraphNodeFindInClone, cuGraphNodeGetContainingGraph, cuGraphNodeGetDependencies, cuGraphNodeGetDependentNodes, cuGraphNodeGetEnabled, cuGraphNodeGetLocalId, cuGraphNodeGetParams, cuGraphNodeGetToolsId, cuGraphNodeGetType, cuGraphNodeSetEnabled, cuGraphNodeSetParams, cuGraphReleaseUserObject, cuGraphRemoveDependencies, cuGraphRetainUserObject, cuGraphUpload, cuGraphicsEGLRegisterImage, cuGraphicsGLRegisterBuffer, cuGraphicsGLRegisterImage, cuGraphicsMapResources, cuGraphicsResourceGetMappedEglFrame, cuGraphicsResourceGetMappedMipmappedArray, cuGraphicsResourceGetMappedPointer, cuGraphicsResourceSetMapFlags, cuGraphicsSubResourceGetMappedArray, cuGraphicsUnmapResources, cuGraphicsUnregisterResource, cuGraphicsVDPAURegisterOutputSurface, cuGraphicsVDPAURegisterVideoSurface

### texture, surface, tensor map (41)

cuSurfObjectCreate, cuSurfObjectDestroy, cuSurfObjectGetResourceDesc, cuSurfRefGetArray, cuSurfRefSetArray, cuTensorMapEncodeIm2col, cuTensorMapEncodeIm2colWide, cuTensorMapEncodeTiled, cuTensorMapReplaceAddress, cuTexObjectCreate, cuTexObjectDestroy, cuTexObjectGetResourceDesc, cuTexObjectGetResourceViewDesc, cuTexObjectGetTextureDesc, cuTexRefCreate, cuTexRefDestroy, cuTexRefGetAddress, cuTexRefGetAddressMode, cuTexRefGetArray, cuTexRefGetBorderColor, cuTexRefGetFilterMode, cuTexRefGetFlags, cuTexRefGetFormat, cuTexRefGetMaxAnisotropy, cuTexRefGetMipmapFilterMode, cuTexRefGetMipmapLevelBias, cuTexRefGetMipmapLevelClamp, cuTexRefGetMipmappedArray, cuTexRefSetAddress, cuTexRefSetAddress2D, cuTexRefSetAddressMode, cuTexRefSetArray, cuTexRefSetBorderColor, cuTexRefSetFilterMode, cuTexRefSetFlags, cuTexRefSetFormat, cuTexRefSetMaxAnisotropy, cuTexRefSetMipmapFilterMode, cuTexRefSetMipmapLevelBias, cuTexRefSetMipmapLevelClamp, cuTexRefSetMipmappedArray

### interop, IPC, VMM handle (38)

cuDestroyExternalMemory, cuEGLApiInit, cuEGLStreamConsumerAcquireFrame, cuEGLStreamConsumerConnect, cuEGLStreamConsumerConnectWithFlags, cuEGLStreamConsumerDisconnect, cuEGLStreamConsumerReleaseFrame, cuEGLStreamProducerConnect, cuEGLStreamProducerDisconnect, cuEGLStreamProducerPresentFrame, cuEGLStreamProducerReturnFrame, cuExternalMemoryGetMappedBuffer, cuExternalMemoryGetMappedMipmappedArray, cuGLCtxCreate, cuGLGetDevices, cuGLInit, cuGLMapBufferObject, cuGLMapBufferObjectAsync, cuGLRegisterBufferObject, cuGLSetBufferObjectMapFlags, cuGLUnmapBufferObject, cuGLUnmapBufferObjectAsync, cuGLUnregisterBufferObject, cuImportExternalMemory, cuImportExternalSemaphore, cuIpcCloseMemHandle, cuIpcGetEventHandle, cuIpcGetMemHandle, cuIpcOpenEventHandle, cuIpcOpenMemHandle, cuMulticastAddDevice, cuMulticastBindAddr, cuMulticastBindMem, cuMulticastCreate, cuMulticastGetGranularity, cuMulticastUnbind, cuVDPAUCtxCreate, cuVDPAUGetDevice

### profiling, debugger, coredump (18)

cuCheckpointOperationComplete, cuCheckpointProcessCheckpoint, cuCheckpointProcessGetRestoreThreadId, cuCheckpointProcessGetState, cuCheckpointProcessLock, cuCheckpointProcessRestore, cuCheckpointProcessUnlock, cuCoredumpDeregisterCompleteCallback, cuCoredumpDeregisterStartCallback, cuCoredumpGetAttribute, cuCoredumpGetAttributeGlobal, cuCoredumpRegisterCompleteCallback, cuCoredumpRegisterStartCallback, cuCoredumpSetAttribute, cuCoredumpSetAttributeGlobal, cuProfilerInitialize, cuProfilerStart, cuProfilerStop

### green context / SM partition (10)

cuDevResourceGenerateDesc, cuDevSmResourceSplit, cuDevSmResourceSplitByCount, cuGreenCtxCreate, cuGreenCtxDestroy, cuGreenCtxGetDevResource, cuGreenCtxGetId, cuGreenCtxRecordEvent, cuGreenCtxStreamCreate, cuGreenCtxWaitEvent

### lain-lain (88)

cuArray3DCreate, cuArray3DGetDescriptor, cuArrayCreate, cuArrayDestroy, cuArrayGetDescriptor, cuArrayGetMemoryRequirements, cuArrayGetPlane, cuArrayGetSparseProperties, cuDriverGetGpuCodeIsaVersion, cuFlushGPUDirectRDMAWrites, cuGetErrorName, **cuGetErrorString**, cuLogicalEndpointAddDevice, cuLogicalEndpointBindAddr, cuLogicalEndpointBindMem, cuLogicalEndpointCreate, cuLogicalEndpointDestroy, cuLogicalEndpointExport, cuLogicalEndpointGetLimits, cuLogicalEndpointIdRelease, cuLogicalEndpointIdReserve, cuLogicalEndpointImport, cuLogicalEndpointQuery, cuLogicalEndpointUnbind, cuLogsCurrent, cuLogsDumpToFile, cuLogsDumpToMemory, cuLogsRegisterCallback, cuLogsUnregisterCallback, cuMemExportToShareableHandle, cuMemGetAccess, cuMemGetAddressRange, cuMemGetAttribute, cuMemGetDefaultMemPool, cuMemGetLocationInfo, cuMemImportFromShareableHandle, cuMemSetMemPool, cuMipmappedArrayCreate, cuMipmappedArrayDestroy, cuMipmappedArrayGetLevel, cuMipmappedArrayGetMemoryRequirements, cuMipmappedArrayGetSparseProperties, cuPointerGetAttribute, cuPointerGetAttributes, cuPointerSetAttribute, cuSubgridCreate, cuSubgridWorkerGridCreate, cuSubgridWorksetCreate, cuThreadExchangeStreamCaptureMode, cuUserObjectCreate, cuUserObjectRelease, cuUserObjectRetain, cudbgApiAttach, cudbgApiClientPid, cudbgApiClientRevision, cudbgApiDetach, cudbgApiInit, cudbgAttachHandlerAvailable, cudbgDebuggerCapabilities, cudbgDebuggerInitialized, cudbgDetachSuspendedDevicesMask, cudbgEnableIntegratedMemcheck, cudbgEnableLaunchBlocking, cudbgEnablePreemptionDebugging, cudbgGetAPI, cudbgGetAPIVersion, cudbgInitiateDebuggerAttachProcedureFd, cudbgInjectionPath, cudbgIpcFlag, cudbgMain, cudbgPreInit, cudbgReportAttachProcedureFinished, cudbgReportDriverApiError, cudbgReportDriverApiErrorFlags, cudbgReportDriverInternalError, cudbgReportedDriverApiErrorCode, cudbgReportedDriverApiErrorFuncNameAddr, cudbgReportedDriverApiErrorFuncNameSize, cudbgReportedDriverApiErrorNameAddr, cudbgReportedDriverApiErrorNameSize, cudbgReportedDriverApiErrorSource, cudbgReportedDriverApiErrorStringAddr, cudbgReportedDriverApiErrorStringSize, cudbgReportedDriverInternalErrorCode, cudbgResumeForAttachDetach, cudbgRpcEnabled, cudbgSessionId, cudbgUseExternalDebugger

Jumlah: 555 nama ternormalisasi dari 762 simbol mentah (varian
`_v2`/`_ptsz`/`_ptds` dilebur).

## Lampiran B — permukaan device

### B.1 Header tipe: jumlah fungsi bernama per header

Angka di bawah hasil memindai deklarasi bermakro (`__CUDA_FP16_DECL__` dan
sejenisnya). Header fp8/fp4/fp6 sebagian besar berupa overload operator, jadi
angkanya kecil meski permukaannya lebar.

| Header | Fungsi | Isi |
|---|---:|---|
| `cuda_fp16.h` | 75 | `__half`/`__half2`: aritmetika, perbandingan, konversi, fungsi matematika (`h2exp`, `h2log`, `h2rcp`, `h2sqrt`), packing |
| `cuda_fp16.hpp` | 81 | implementasi `__half` (sama dengan di atas, ditambah helper internal) |
| `cuda_bf16.h` | 92 | `__nv_bfloat16`/`__nv_bfloat162`: keluarga yang sama untuk bf16 |
| `cuda_bf16.hpp` | 136 | implementasi bf16 |
| `cuda_fp8.hpp` | 4 | fp8 e4m3/e5m2: `__nv_cvt_bfloat16raw_to_fp8`, `__nv_cvt_halfraw2_to_fp8x2`, sisanya operator |
| `cuda_fp4.hpp` | 2 | fp4: `__nv_cvt_bfloat16raw_to_fp4`, `__nv_cvt_halfraw2_to_fp4x2` + operator |
| `cuda_fp6.hpp` | 2 | fp6: `__nv_cvt_bfloat16raw_to_fp6`, `__nv_cvt_halfraw2_to_fp6x2` + operator |

### B.2 Header intrinsik: nama lengkap

**`sm_20_intrinsics.h`** — 70 deklarasi

`__dadd_rd`, `__dadd_rn`, `__dadd_ru`, `__dadd_rz`, `__ddiv_rd`, `__ddiv_rn`, `__ddiv_ru`, `__ddiv_rz`, `__dmul_rd`, `__dmul_rn`, `__dmul_ru`, `__dmul_rz`, `__double2float_rd`, `__double2float_rn`, `__double2float_ru`, `__double2float_rz`, `__double2hiint`, `__double2int_rd`, `__double2int_rn`, `__double2int_ru`, `__double2ll_rd`, `__double2ll_rn`, `__double2ll_ru`, `__double2loint`, `__double2uint_rd`, `__double2uint_rn`, `__double2uint_ru`, `__double2ull_rd`, `__double2ull_rn`, `__double2ull_ru`, `__double_as_longlong`, `__drcp_rd`, `__drcp_rn`, `__drcp_ru`, `__drcp_rz`, `__dsqrt_rd`, `__dsqrt_rn`, `__dsqrt_ru`, `__dsqrt_rz`, `__dsub_rd`, `__dsub_rn`, `__dsub_ru`, `__dsub_rz`, `__fma_rd`, `__fma_rn`, `__fma_ru`, `__fma_rz`, `__fmaf_ieee_rd`, `__fmaf_ieee_rn`, `__fmaf_ieee_ru`, `__fmaf_ieee_rz`, `__hiloint2double`, `__int2double_rn`, `__ll2double_rd`, `__ll2double_rn`, `__ll2double_ru`, `__ll2double_rz`, `__longlong_as_double`, `__syncthreads_and`, `__syncthreads_count`, `__syncthreads_or`, `__threadfence_system`, `__uint2double_rn`, `__ull2double_rd`, `__ull2double_rn`, `__ull2double_ru`, `__ull2double_rz`, `clock64`

**`sm_30_intrinsics.hpp`** — 14 deklarasi

`__nvvm_bar_warp_sync`, `__nvvm_barrier_sync`, `__nvvm_barrier_sync_cnt`, `__nvvm_fns`, `__nvvm_shfl_bfly_sync`, `__nvvm_shfl_down_sync`, `__nvvm_shfl_idx_sync`, `__nvvm_shfl_up_sync`, `__nvvm_vote_all_sync`, `__nvvm_vote_any_sync`, `__nvvm_vote_ballot_sync`, `__nvvm_vote_uni_sync`

**`sm_32_intrinsics.hpp`** — 35 deklarasi



**`sm_61_intrinsics.hpp`** — 7 deklarasi



**`device_atomic_functions.hpp`** — 23 deklarasi

`__fAtomicExch`, `__iAtomicAdd`, `__iAtomicAnd`, `__iAtomicCAS`, `__iAtomicExch`, `__iAtomicMax`, `__iAtomicMin`, `__iAtomicOr`, `__iAtomicXor`, `__uAtomicAdd`, `__uAtomicAnd`, `__uAtomicCAS`, `__uAtomicDec`, `__uAtomicExch`, `__uAtomicInc`, `__uAtomicMax`, `__uAtomicMin`, `__uAtomicOr`, `__uAtomicXor`, `__ullAtomicAdd`, `__ullAtomicCAS`, `__ullAtomicExch`

**`sm_60_atomic_functions.hpp`** — 70 deklarasi

`__dAtomicAdd`, `__dAtomicAdd_block`, `__dAtomicAdd_system`, `__fAtomicAdd_block`, `__fAtomicAdd_system`, `__fAtomicExch_block`, `__fAtomicExch_system`, `__iAtomicAdd_block`, `__iAtomicAdd_system`, `__iAtomicAnd_block`, `__iAtomicAnd_system`, `__iAtomicCAS_block`, `__iAtomicCAS_system`, `__iAtomicExch_block`, `__iAtomicExch_system`, `__iAtomicMax_block`, `__iAtomicMax_system`, `__iAtomicMin_block`, `__iAtomicMin_system`, `__iAtomicOr_block`, `__iAtomicOr_system`, `__iAtomicXor_block`, `__iAtomicXor_system`, `__illAtomicMax_block`, `__illAtomicMax_system`, `__illAtomicMin_block`, `__illAtomicMin_system`, `__llAtomicAnd_block`, `__llAtomicAnd_system`, `__llAtomicOr_block`, `__llAtomicOr_system`, `__llAtomicXor_block`, `__llAtomicXor_system`, `__uAtomicAdd_block`, `__uAtomicAdd_system`, `__uAtomicAnd_block`, `__uAtomicAnd_system`, `__uAtomicCAS_block`, `__uAtomicCAS_system`, `__uAtomicDec_block`, `__uAtomicDec_system`, `__uAtomicExch_block`, `__uAtomicExch_system`, `__uAtomicInc_block`, `__uAtomicInc_system`, `__uAtomicMax_block`, `__uAtomicMax_system`, `__uAtomicMin_block`, `__uAtomicMin_system`, `__uAtomicOr_block`, `__uAtomicOr_system`, `__uAtomicXor_block`, `__uAtomicXor_system`, `__ullAtomicAdd_block`, `__ullAtomicAdd_system`, `__ullAtomicAnd_block`, `__ullAtomicAnd_system`, `__ullAtomicCAS_block`, `__ullAtomicCAS_system`, `__ullAtomicExch_block`, `__ullAtomicExch_system`, `__ullAtomicMax_block`, `__ullAtomicMax_system`, `__ullAtomicMin_block`, `__ullAtomicMin_system`, `__ullAtomicOr_block`, `__ullAtomicOr_system`, `__ullAtomicXor_block`, `__ullAtomicXor_system`

**`cuda_pipeline_helpers.h`** — 2 deklarasi

`__nvvm_get_smem_pointer`

**`cuda_awbarrier_helpers.h`** — 2 deklarasi

`__nvvm_get_smem_pointer`

**`cuda_device_runtime_api.h`** — 108 deklarasi

`__CDPRT_DEPRECATED`, `__cudaCDP2DeviceGetAttribute`, `__cudaCDP2DeviceGetCacheConfig`, `__cudaCDP2DeviceGetLimit`, `__cudaCDP2DeviceGetSharedMemConfig`, `__cudaCDP2EventCreateWithFlags`, `__cudaCDP2EventDestroy`, `__cudaCDP2EventRecord`, `__cudaCDP2EventRecordWithFlags`, `__cudaCDP2EventRecordWithFlags_ptsz`, `__cudaCDP2EventRecord_ptsz`, `__cudaCDP2Free`, `__cudaCDP2FuncGetAttributes`, `__cudaCDP2GetDevice`, `__cudaCDP2GetDeviceCount`, `__cudaCDP2GetErrorName`, `__cudaCDP2GetErrorString`, `__cudaCDP2GetLastError`, `__cudaCDP2GetParameterBuffer`, `__cudaCDP2GetParameterBufferV2`, `__cudaCDP2LaunchDevice`, `__cudaCDP2LaunchDeviceV2`, `__cudaCDP2LaunchDeviceV2_ptsz`, `__cudaCDP2LaunchDevice_ptsz`, `__cudaCDP2Malloc`, `__cudaCDP2Memcpy2DAsync`, `__cudaCDP2Memcpy2DAsync_ptsz`, `__cudaCDP2Memcpy3DAsync`, `__cudaCDP2Memcpy3DAsync_ptsz`, `__cudaCDP2MemcpyAsync`, `__cudaCDP2MemcpyAsync_ptsz`, `__cudaCDP2Memset2DAsync`, `__cudaCDP2Memset2DAsync_ptsz`, `__cudaCDP2Memset3DAsync`, `__cudaCDP2Memset3DAsync_ptsz`, `__cudaCDP2MemsetAsync`, `__cudaCDP2MemsetAsync_ptsz`, `__cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessor`, `__cudaCDP2OccupancyMaxActiveBlocksPerMultiprocessorWithFlags`, `__cudaCDP2PeekAtLastError`, `__cudaCDP2RuntimeGetVersion`, `__cudaCDP2StreamCreateWithFlags`, `__cudaCDP2StreamDestroy`, `__cudaCDP2StreamWaitEvent`, `__cudaCDP2StreamWaitEvent_ptsz`, `__cudaDeviceSynchronizeDeprecationAvoidance`, `cudaCGGetIntrinsicHandle`, `cudaCGGetRank`, `cudaCGGetSize`, `cudaCGSynchronize`, `cudaCGSynchronizeGrid`, `cudaDeviceGetAttribute`, `cudaDeviceGetCacheConfig`, `cudaDeviceGetLimit`, `cudaDeviceGetSharedMemConfig`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventRecordWithFlags`, `cudaEventRecordWithFlags_ptsz`, `cudaEventRecord_ptsz`, `cudaFabricOpErrorStatusCount`, `cudaFabricOpErrorStatusGet`, `cudaFree`, `cudaFuncGetAttributes`, `cudaGetCurrentGraphExec`, `cudaGetDevice`, `cudaGetDeviceCount`, `cudaGetErrorName`, `cudaGetErrorString`, `cudaGetLastError`, `cudaGetParameterBuffer`, `cudaGetParameterBufferV2`, `cudaGraphKernelNodeSetEnabled`, `cudaGraphKernelNodeSetGridDim`, `cudaGraphKernelNodeSetParam`, `cudaGraphKernelNodeUpdatesApply`, `cudaGraphLaunch`, `cudaGraphLaunch_ptsz`, `cudaGraphSetConditional`, `cudaGridDependencySynchronize`, `cudaLaunchDevice`, `cudaLaunchDeviceV2`, `cudaLaunchDeviceV2_ptsz`, `cudaLaunchDevice_ptsz`, `cudaMalloc`, `cudaMemcpy2DAsync`, `cudaMemcpy2DAsync_ptsz`, `cudaMemcpy3DAsync`, `cudaMemcpy3DAsync_ptsz`, `cudaMemcpyAsync`, `cudaMemcpyAsync_ptsz`, `cudaMemset2DAsync`, `cudaMemset2DAsync_ptsz`, `cudaMemset3DAsync`, `cudaMemset3DAsync_ptsz`, `cudaMemsetAsync`, `cudaMemsetAsync_ptsz`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`, `cudaPeekAtLastError`, `cudaRuntimeGetVersion`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamGraphTailLaunch`, `cudaStreamWaitEvent`, `cudaStreamWaitEvent_ptsz`, `cudaTriggerProgrammaticLaunchCompletion`

**`cooperative_groups.h`** — 3 deklarasi

`multi_warp_collectives_helper`, `multi_warp_scratch_location_getter`, `multi_warp_sync_location_getter`

### B.3 libcu++: 63 pembungkus instruksi PTX

Satu file per instruksi di `cccl/cuda/__ptx/instructions/`. Ini daftar
instruksi PTX yang bisa dipanggil dari C++ tanpa menulis `asm` sendiri.

`applypriority_async_bulk`, `barrier_cluster`, `bfind`, `bmsk`, `clusterlaunchcontrol`, `cp_async_bulk`, `cp_async_bulk_commit_group`, `cp_async_bulk_prefetch`, `cp_async_bulk_prefetch_tensor`, `cp_async_bulk_tensor`, `cp_async_bulk_wait_group`, `cp_async_mbarrier_arrive`, `cp_async_mbarrier_arrive_noinc`, `cp_reduce_async_bulk`, `cp_reduce_async_bulk_tensor`, `elect_sync`, `exit`, `fabric_submit`, `fabric_try_get`, `fabric_try_pullred`, `fabric_try_put`, `fabric_try_red`, `fabric_wait`, `fence`, `get_sreg`, `getctarank`, `ld`, `ldmatrix`, `mbarrier_arrive`, `mbarrier_check_layout`, `mbarrier_complete_tx`, `mbarrier_expect_tx`, `mbarrier_init`, `mbarrier_inval`, `mbarrier_pending_count`, `mbarrier_wait`, `multimem_ld_reduce`, `multimem_red`, `multimem_st`, `prefetch`, `prmt`, `red_async`, `setmaxnreg`, `shfl_sync`, `shl`, `shr`, `st`, `st_async`, `st_bulk`, `stmatrix`, `tcgen05_alloc`, `tcgen05_commit`, `tcgen05_cp`, `tcgen05_fence`, `tcgen05_ld`, `tcgen05_mma`, `tcgen05_mma_ws`, `tcgen05_shift`, `tcgen05_st`, `tcgen05_wait`, `tensormap_cp_fenceproxy`, `tensormap_replace`, `trap`

Untuk `sm_89`, yang sudah diuji di sini (bagian 2) dan lolos: `ld`, `st`,
`ldmatrix`, `mbarrier.init`/`mbarrier.arrive`, `prefetch`, `cp_async_mbarrier_arrive`.
Yang sudah diuji dan gagal: `cp_async_bulk*`, `stmatrix`, `elect_sync` (semua
`sm_90+`). Sisanya belum diuji satu per satu di sini — `tcgen05_*` (`sm_100a`),
`clusterlaunchcontrol`, `multimem_*`, `tensormap_*`, `fabric_*`, `setmaxnreg`
dan `red_async`/`st_async` jelas di luar jangkauan `sm_89`, sedangkan
`applypriority_async_bulk`, `bfind`, `bmsk`, `prmt`, `shl`, `shr`, `get_sreg`,
`fence`, `trap`, `exit` perlu probe sendiri sebelum dipakai. Pola probe-nya di
bagian 5; jangan percaya daftar ini tanpa menjalankannya.
