use crate::collect::CollectStatus;
use crate::collect::win::OwnedLibrary;
use std::ffi::{c_char, c_void};

use super::{clamp_percent, power_percent, push_history};

const NVAPI_OK: i32 = 0;
const NVAPI_MAX_PHYSICAL_GPUS: usize = 64;
const NVAPI_SHORT_STRING_MAX: usize = 64;

/// Graphics clock domain index in `NvClockFrequencies`.
const CLOCK_DOMAIN_GRAPHICS: usize = 0;

/// NvAPI clock type: current frequency.
const CLOCK_TYPE_CURRENT: u32 = 0;
/// NvAPI clock type: boost frequency (used for max clock).
const CLOCK_TYPE_BOOST: u32 = 2;

/// Power topology domain representing total board power.
const POWER_DOMAIN_BOARD: u32 = 0;

/// Per Cent Mille base: 100,000 PCM = 100%.
const PCM_100_PERCENT: u64 = 100_000;

/// Thermal target for the GPU die.
const NVAPI_THERMAL_TARGET_GPU: i32 = 1;
/// Request all thermal sensors.
const NVAPI_THERMAL_TARGET_ALL: u32 = 15;

const NVML_SUCCESS: u32 = 0;
/// `NVML_DEVICE_UUID_BUFFER_SIZE` from NVML headers — buffer size
/// (including null terminator) sufficient to receive any GPU UUID
/// string `nvmlDeviceGetUUID` produces (`GPU-...` form, ~40 chars).
const NVML_DEVICE_UUID_BUFFER_SIZE: usize = 80;

type NvmlDevice = *mut c_void;
type NvmlInitV2 = unsafe extern "C" fn() -> u32;
type NvmlShutdownFn = unsafe extern "C" fn() -> u32;
type NvmlDeviceGetCountV2 = unsafe extern "C" fn(*mut u32) -> u32;
type NvmlDeviceGetHandleByIndexV2 = unsafe extern "C" fn(u32, *mut NvmlDevice) -> u32;
type NvmlDeviceGetPowerManagementDefaultLimit = unsafe extern "C" fn(NvmlDevice, *mut u32) -> u32;
type NvmlDeviceGetUuid = unsafe extern "C" fn(NvmlDevice, *mut c_char, u32) -> u32;
type NvmlDeviceGetUtilizationRates = unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> u32;
type NvmlDeviceGetTemperature =
    unsafe extern "C" fn(NvmlDevice, NvmlTemperatureSensor, *mut u32) -> u32;
type NvmlDeviceGetPowerUsage = unsafe extern "C" fn(NvmlDevice, *mut u32) -> u32;
type NvmlDeviceGetClockInfo = unsafe extern "C" fn(NvmlDevice, NvmlClockType, *mut u32) -> u32;
type NvmlDeviceGetMaxClockInfo = unsafe extern "C" fn(NvmlDevice, NvmlClockType, *mut u32) -> u32;

/// `NVML_TEMPERATURE_GPU` — the GPU die sensor.
const NVML_TEMPERATURE_GPU: NvmlTemperatureSensor = 0;
/// `NVML_CLOCK_GRAPHICS` — the graphics/SM clock domain.
const NVML_CLOCK_GRAPHICS: NvmlClockType = 0;

/// `nvmlTemperatureSensors_t` selector, as passed to
/// `nvmlDeviceGetTemperature`.
type NvmlTemperatureSensor = u32;
/// `nvmlClockType_t` selector, as passed to the clock queries.
type NvmlClockType = u32;

/// NVML utilization sample, in percent. Written by
/// `nvmlDeviceGetUtilizationRates`; `memory` is the VRAM controller
/// occupancy, which rtop does not currently render.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvmlUtilization {
    gpu: u32,
    memory: u32,
}

type NvPhysicalGpuHandle = *mut c_void;

/// GPU utilization domain (one per metric type).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvUtilDomain {
    /// Bit 0: domain is present.
    flags: u32,
    /// Utilization percentage (0–100).
    percentage: u32,
}

#[repr(C)]
struct NvDynamicPstatesInfoEx {
    version: u32,
    flags: u32,
    utilizations: [NvUtilDomain; 8],
}

/// Single thermal sensor reading.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvThermalSensor {
    controller: i32,
    default_min_temp: i32,
    default_max_temp: i32,
    current_temp: i32,
    target: i32,
}

/// GPU thermal settings (up to 3 sensors).
#[repr(C)]
struct NvThermalSettings {
    version: u32,
    count: u32,
    sensors: [NvThermalSensor; 3],
}

/// Display driver memory info (V3). All size fields are in KB.
#[repr(C)]
struct NvMemoryInfo {
    version: u32,
    dedicated_video_memory_kb: u32,
    avail_dedicated_video_memory_kb: u32,
    system_video_memory_kb: u32,
    shared_system_memory_kb: u32,
    cur_avail_dedicated_video_memory_kb: u32,
    dedicated_video_memory_evictions_size_kb: u32,
    dedicated_video_memory_eviction_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvClockDomain {
    /// Bit 0: domain is present.
    flags: u32,
    /// Frequency in kHz.
    frequency_khz: u32,
}

/// GPU clock frequencies (up to 32 domains).
#[repr(C)]
struct NvClockFrequencies {
    version: u32,
    clock_type: u32,
    domains: [NvClockDomain; 32],
}

/// Single power topology entry (undocumented NvAPI).
/// Values are in Per Cent Mille (PCM): 100,000 = 100% of TDP.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvPowerTopoEntry {
    domain: u32,
    flags: u32,
    power_pcm: u32,
    unknown: u32,
}

/// Power topology status (up to 4 entries, undocumented NvAPI).
#[repr(C)]
struct NvPowerTopo {
    version: u32,
    count: u32,
    entries: [NvPowerTopoEntry; 4],
}

/// Single power info entry (undocumented NvAPI). Values are in PCM.
/// Fields at known offsets hold min/default/max power as a percentage
/// of TDP (100,000 PCM = 100%); intervening fields are reserved.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NvPowerInfoEntry {
    pstate: u32,
    reserved_0: [u32; 2],
    min_power_pcm: u32,
    reserved_1: [u32; 2],
    def_power_pcm: u32,
    reserved_2: [u32; 2],
    max_power_pcm: u32,
    reserved_3: u32,
}

/// Power policies info (up to 4 entries, undocumented NvAPI).
#[repr(C)]
struct NvPowerInfo {
    version: u32,
    valid: u8,
    count: u8,
    padding: [u8; 2],
    entries: [NvPowerInfoEntry; 4],
}

/// NvAPI struct version: `sizeof(T) | (ver << 16)`.
const fn nvapi_version<T>(ver: u32) -> u32 {
    std::mem::size_of::<T>() as u32 | (ver << 16)
}

const PSTATES_INFO_VER: u32 = nvapi_version::<NvDynamicPstatesInfoEx>(1);
const THERMAL_SETTINGS_VER: u32 = nvapi_version::<NvThermalSettings>(2);
const MEMORY_INFO_VER: u32 = nvapi_version::<NvMemoryInfo>(3);
const CLOCK_FREQUENCIES_VER: u32 = nvapi_version::<NvClockFrequencies>(3);
const POWER_TOPO_VER: u32 = nvapi_version::<NvPowerTopo>(1);
const POWER_INFO_VER: u32 = nvapi_version::<NvPowerInfo>(1);

type NvApiInitialize = unsafe extern "C" fn() -> i32;
type NvApiUnload = unsafe extern "C" fn() -> i32;
type NvApiEnumPhysicalGPUs =
    unsafe extern "C" fn(handles: *mut NvPhysicalGpuHandle, count: *mut u32) -> i32;
type NvApiGpuGetFullName =
    unsafe extern "C" fn(handle: NvPhysicalGpuHandle, name: *mut c_char) -> i32;
type NvApiGetDynamicPstatesInfoEx =
    unsafe extern "C" fn(NvPhysicalGpuHandle, *mut NvDynamicPstatesInfoEx) -> i32;
type NvApiGetThermalSettings =
    unsafe extern "C" fn(NvPhysicalGpuHandle, u32, *mut NvThermalSettings) -> i32;
type NvApiGetMemoryInfo = unsafe extern "C" fn(NvPhysicalGpuHandle, *mut NvMemoryInfo) -> i32;
type NvApiGetAllClockFrequencies =
    unsafe extern "C" fn(NvPhysicalGpuHandle, *mut NvClockFrequencies) -> i32;
type NvApiClientPowerTopoGetStatus =
    unsafe extern "C" fn(NvPhysicalGpuHandle, *mut NvPowerTopo) -> i32;
type NvApiClientPowerPoliciesGetInfo =
    unsafe extern "C" fn(NvPhysicalGpuHandle, *mut NvPowerInfo) -> i32;

const ID_INITIALIZE: u32 = 0x0150_e828;
const ID_UNLOAD: u32 = 0xd22b_dd7e;
const ID_ENUM_PHYSICAL_GPUS: u32 = 0xe5ac_921f;
const ID_GPU_GET_FULL_NAME: u32 = 0xceee_8e9f;
const ID_GPU_GET_DYNAMIC_PSTATES_INFO_EX: u32 = 0x60de_d2ed;
const ID_GPU_GET_THERMAL_SETTINGS: u32 = 0xe364_0a56;
const ID_GPU_GET_MEMORY_INFO: u32 = 0x07f9_b368;
const ID_GPU_GET_ALL_CLOCK_FREQUENCIES: u32 = 0xdcb6_16c3;
const ID_GPU_CLIENT_POWER_TOPO_GET_STATUS: u32 = 0xedcf_624e;
const ID_GPU_CLIENT_POWER_POLICIES_GET_INFO: u32 = 0x3420_6d86;

/// A device as NVML sees it, paired with the static metadata rtop
/// binds once at discovery. The handle stays valid until
/// `nvmlShutdown`, so it outlives every per-cycle query.
#[derive(Clone)]
struct NvmlDeviceEntry {
    handle: NvmlDevice,
    /// Default power management limit (TDP) in milliwatts. `0` when
    /// NVML did not return a value for this device — the caller
    /// treats `0` as "no TDP info available".
    tdp_mw: u32,
    /// `nvmlDeviceGetUUID` output (e.g. `GPU-12345678-...`). Empty
    /// string when the call failed; the caller's stable-id fallback
    /// path handles the empty case without losing the device.
    uuid: String,
}

/// Resolved `nvml.dll` function-pointer table.
///
/// NVML is the primary metric source on Windows: driver 570+ reports
/// `NVAPI_NOT_SUPPORTED` for the legacy NvAPI entry points rtop
/// originally used for utilization, temperature, power, and clocks,
/// while every function resolved here is still served. Symbols whose
/// absence is tolerable (currently only the max-clock query) resolve
/// to `None` instead of failing the whole load.
struct NvmlFunctions {
    _library: OwnedLibrary,
    init: NvmlInitV2,
    shutdown: NvmlShutdownFn,
    get_count: NvmlDeviceGetCountV2,
    get_handle: NvmlDeviceGetHandleByIndexV2,
    get_default_limit: NvmlDeviceGetPowerManagementDefaultLimit,
    get_uuid: NvmlDeviceGetUuid,
    get_utilization_rates: NvmlDeviceGetUtilizationRates,
    get_temperature: NvmlDeviceGetTemperature,
    get_power_usage: NvmlDeviceGetPowerUsage,
    get_clock_info: NvmlDeviceGetClockInfo,
    /// Optional — absent on NVML builds predating the query.
    get_max_clock_info: Option<NvmlDeviceGetMaxClockInfo>,
}

impl NvmlFunctions {
    /// Load `nvml.dll` and resolve the function table without
    /// initialising NVML. Returns `None` when the library or a
    /// required symbol is missing.
    fn load() -> Option<Self> {
        use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
        use windows::core::PCSTR;

        let dll_name: Vec<u16> = "nvml.dll\0".encode_utf16().collect();
        // SAFETY: LoadLibraryW receives a valid null-terminated UTF-16 DLL name.
        // The returned handle is checked via ok()? before use.
        let library = OwnedLibrary::new(
            unsafe { LoadLibraryW(windows::core::PCWSTR(dll_name.as_ptr())) }.ok()?,
        )?;
        let handle = library.get();

        /// Resolve a required NVML symbol; a missing one aborts the load.
        macro_rules! load_fn {
            ($name:literal, $ty:ty) => {{
                // SAFETY: handle is a valid loaded nvml.dll module.
                let proc = unsafe { GetProcAddress(handle, PCSTR(concat!($name, "\0").as_ptr())) }?;
                // SAFETY: GetProcAddress returned a non-null address for a known
                // NVML symbol; the target type matches the documented signature.
                unsafe { std::mem::transmute::<unsafe extern "system" fn() -> isize, $ty>(proc) }
            }};
        }

        /// Resolve an optional NVML symbol; a missing one yields `None`
        /// and costs only the metrics that depend on it.
        macro_rules! load_fn_opt {
            ($name:literal, $ty:ty) => {{
                // SAFETY: handle is a valid loaded nvml.dll module.
                let proc = unsafe { GetProcAddress(handle, PCSTR(concat!($name, "\0").as_ptr())) };
                // SAFETY: when non-null, GetProcAddress returned the address of
                // the named NVML symbol and the target type matches its
                // documented signature.
                proc.map(|p| unsafe {
                    std::mem::transmute::<unsafe extern "system" fn() -> isize, $ty>(p)
                })
            }};
        }

        Some(Self {
            init: load_fn!("nvmlInit_v2", NvmlInitV2),
            shutdown: load_fn!("nvmlShutdown", NvmlShutdownFn),
            get_count: load_fn!("nvmlDeviceGetCount_v2", NvmlDeviceGetCountV2),
            get_handle: load_fn!(
                "nvmlDeviceGetHandleByIndex_v2",
                NvmlDeviceGetHandleByIndexV2
            ),
            get_default_limit: load_fn!(
                "nvmlDeviceGetPowerManagementDefaultLimit",
                NvmlDeviceGetPowerManagementDefaultLimit
            ),
            get_uuid: load_fn!("nvmlDeviceGetUUID", NvmlDeviceGetUuid),
            get_utilization_rates: load_fn!(
                "nvmlDeviceGetUtilizationRates",
                NvmlDeviceGetUtilizationRates
            ),
            get_temperature: load_fn!("nvmlDeviceGetTemperature", NvmlDeviceGetTemperature),
            get_power_usage: load_fn!("nvmlDeviceGetPowerUsage", NvmlDeviceGetPowerUsage),
            get_clock_info: load_fn!("nvmlDeviceGetClockInfo", NvmlDeviceGetClockInfo),
            get_max_clock_info: load_fn_opt!(
                "nvmlDeviceGetMaxClockInfo",
                NvmlDeviceGetMaxClockInfo
            ),
            _library: library,
        })
    }
}

/// Live NVML session: an initialised [`NvmlFunctions`] table whose
/// device handles stay usable for the collector's lifetime.
///
/// Owned once per vendor by [`super::GpuCollector`] (via
/// [`NvApiBundle`]) and dropped after the per-device entries, which
/// hold the handles — the `Drop` impl calls `nvmlShutdown` exactly
/// once per successful `nvmlInit_v2`.
pub(super) struct NvmlSession {
    nvml: NvmlFunctions,
}

impl NvmlSession {
    /// Load `nvml.dll` and call `nvmlInit_v2`. Returns `None` when the
    /// library is absent, a required symbol is missing, or
    /// initialisation fails — in every case the caller falls back
    /// to the NvAPI path for all metrics.
    fn load() -> Option<Self> {
        let nvml = NvmlFunctions::load()?;

        // SAFETY: nvml.init was resolved from nvml.dll and matches the
        // documented nvmlInit_v2 signature.
        let ret = unsafe { (nvml.init)() };
        if ret != NVML_SUCCESS {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvml,
                code = %crate::log::Hex(ret),
                "nvmlInit_v2 failed",
            );
            return None;
        }
        Some(Self { nvml })
    }

    /// Enumerate NVML devices with their static metadata, padded to
    /// exactly `want` entries with `None` so the result can be zipped
    /// 1:1 with the NvAPI device list.
    ///
    /// Index correspondence between NVML and NvAPI is documented as
    /// "best-effort": both APIs typically enumerate in the same
    /// driver-defined order (PCI bus). The TDP wiring has relied on
    /// this ordering since rtop's first NVIDIA support; the UUID
    /// wiring inherits the same assumption.
    fn devices(&self, want: usize) -> Vec<Option<NvmlDeviceEntry>> {
        let nvml = &self.nvml;
        let mut count: u32 = 0;
        // SAFETY: count is a valid pointer to a stack-allocated u32.
        let _ = unsafe { (nvml.get_count)(&mut count) };

        let limit = (count as usize).min(want);
        let mut entries: Vec<Option<NvmlDeviceEntry>> = Vec::with_capacity(want);
        for i in 0..limit as u32 {
            let mut handle: NvmlDevice = std::ptr::null_mut();
            // SAFETY: i is within the reported count; handle is a valid
            // pointer for the call to write through.
            if unsafe { (nvml.get_handle)(i, &mut handle) } != NVML_SUCCESS || handle.is_null() {
                entries.push(None);
                continue;
            }

            let mut limit_mw: u32 = 0;
            // SAFETY: handle is a valid device handle; limit_mw is a valid pointer.
            let tdp_ret = unsafe { (nvml.get_default_limit)(handle, &mut limit_mw) };
            let tdp_mw = if tdp_ret == NVML_SUCCESS {
                tracing::debug!(
                    subsystem = %crate::log::Subsystem::GpuNvml,
                    device = i,
                    limit_mw,
                    "default power limit read",
                );
                limit_mw
            } else {
                tracing::debug!(
                    subsystem = %crate::log::Subsystem::GpuNvml,
                    device = i,
                    code = %crate::log::Hex(tdp_ret),
                    "GetPowerManagementDefaultLimit failed",
                );
                0
            };

            let mut uuid_buf = [0u8; NVML_DEVICE_UUID_BUFFER_SIZE];
            // SAFETY: handle is a valid device handle; uuid_buf is a valid
            // mutable buffer with exactly NVML_DEVICE_UUID_BUFFER_SIZE bytes.
            let uuid_ret = unsafe {
                (nvml.get_uuid)(
                    handle,
                    uuid_buf.as_mut_ptr() as *mut c_char,
                    NVML_DEVICE_UUID_BUFFER_SIZE as u32,
                )
            };
            let uuid = if uuid_ret == NVML_SUCCESS {
                crate::collect::win::string_from_c_buf(&uuid_buf)
            } else {
                tracing::debug!(
                    subsystem = %crate::log::Subsystem::GpuNvml,
                    device = i,
                    code = %crate::log::Hex(uuid_ret),
                    "nvmlDeviceGetUUID failed",
                );
                String::new()
            };

            entries.push(Some(NvmlDeviceEntry {
                handle,
                tdp_mw,
                uuid,
            }));
        }
        entries.resize(want, None);
        entries
    }
}

impl Drop for NvmlSession {
    fn drop(&mut self) {
        // SAFETY: self.nvml.shutdown was resolved from nvml.dll and is
        // called exactly once per successful `nvmlInit_v2` — the
        // session is constructed only when init returns NVML_SUCCESS.
        unsafe {
            let _ = (self.nvml.shutdown)();
        }
    }
}

/// Convert a Per Cent Mille value to milliwatts using a TDP reference.
fn pcm_to_mw(pcm: u64, tdp_mw: u64) -> u64 {
    pcm.saturating_mul(tdp_mw) / PCM_100_PERCENT
}

struct NvApiFunctions {
    _library: OwnedLibrary,
    initialize: NvApiInitialize,
    unload: NvApiUnload,
    enum_physical_gpus: NvApiEnumPhysicalGPUs,
    gpu_get_full_name: NvApiGpuGetFullName,
    get_dynamic_pstates_info_ex: NvApiGetDynamicPstatesInfoEx,
    get_thermal_settings: NvApiGetThermalSettings,
    get_memory_info: NvApiGetMemoryInfo,
    get_all_clock_frequencies: NvApiGetAllClockFrequencies,
    client_power_topo_get_status: NvApiClientPowerTopoGetStatus,
    client_power_policies_get_info: NvApiClientPowerPoliciesGetInfo,
}

impl NvApiFunctions {
    fn load() -> Option<Self> {
        use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
        use windows::core::PCSTR;

        let dll_name: Vec<u16> = "nvapi64.dll\0".encode_utf16().collect();
        // SAFETY: LoadLibraryW receives a valid null-terminated UTF-16 DLL name.
        // The returned handle is checked via ok()? before use.
        let library = OwnedLibrary::new(
            unsafe { LoadLibraryW(windows::core::PCWSTR(dll_name.as_ptr())) }.ok()?,
        )?;

        // NvAPI exports a single symbol: nvapi_QueryInterface.
        // All other functions are resolved by numeric ID through it.
        let handle = library.get();
        // SAFETY: handle is a valid loaded nvapi64.dll module. The symbol name
        // is a static null-terminated byte string.
        let query_interface: unsafe extern "C" fn(u32) -> *mut c_void = unsafe {
            let proc = GetProcAddress(handle, PCSTR(c"nvapi_QueryInterface".as_ptr().cast()))?;
            std::mem::transmute::<unsafe extern "system" fn() -> isize, _>(proc)
        };

        /// Resolve an NvAPI function pointer by ID. Returns `None` if the
        /// function is not present in this driver version.
        macro_rules! resolve {
            ($id:expr, $ty:ty) => {{
                // SAFETY: query_interface was obtained from a valid nvapi64.dll
                // export. The ID is a known NvAPI function identifier.
                let ptr = unsafe { query_interface($id) };
                if ptr.is_null() {
                    return None;
                }
                // SAFETY: query_interface returned a non-null pointer for a
                // known NvAPI function ID. The target type matches the
                // documented NvAPI signature for this ID.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) }
            }};
        }

        Some(Self {
            _library: library,
            initialize: resolve!(ID_INITIALIZE, NvApiInitialize),
            unload: resolve!(ID_UNLOAD, NvApiUnload),
            enum_physical_gpus: resolve!(ID_ENUM_PHYSICAL_GPUS, NvApiEnumPhysicalGPUs),
            gpu_get_full_name: resolve!(ID_GPU_GET_FULL_NAME, NvApiGpuGetFullName),
            get_dynamic_pstates_info_ex: resolve!(
                ID_GPU_GET_DYNAMIC_PSTATES_INFO_EX,
                NvApiGetDynamicPstatesInfoEx
            ),
            get_thermal_settings: resolve!(ID_GPU_GET_THERMAL_SETTINGS, NvApiGetThermalSettings),
            get_memory_info: resolve!(ID_GPU_GET_MEMORY_INFO, NvApiGetMemoryInfo),
            get_all_clock_frequencies: resolve!(
                ID_GPU_GET_ALL_CLOCK_FREQUENCIES,
                NvApiGetAllClockFrequencies
            ),
            client_power_topo_get_status: resolve!(
                ID_GPU_CLIENT_POWER_TOPO_GET_STATUS,
                NvApiClientPowerTopoGetStatus
            ),
            client_power_policies_get_info: resolve!(
                ID_GPU_CLIENT_POWER_POLICIES_GET_INFO,
                NvApiClientPowerPoliciesGetInfo
            ),
        })
    }
}

/// `Send + Sync` newtype around `NvPhysicalGpuHandle` (`*mut c_void`).
///
/// NvAPI handles are opaque identifiers that NvAPI dereferences
/// internally; the rtop side never reads through them. NvAPI itself
/// is documented thread-safe per `nvapi.h`, so a handle owned by
/// one device thread can be passed back to NvAPI from that same
/// thread without external synchronisation. The wrapper exists
/// purely to satisfy the `Send + Sync` requirement on the per-device
/// collector type that crosses a thread boundary in
/// [`crate::runner::spawn_collector`].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(super) struct SendNvHandle(NvPhysicalGpuHandle);

// SAFETY: NvAPI handles are opaque identifiers used only by NvAPI
// itself, which is thread-safe per nvapi.h. We never dereference the
// pointer from Rust.
unsafe impl Send for SendNvHandle {}
// SAFETY: see Send impl above.
unsafe impl Sync for SendNvHandle {}

/// NvAPI vendor session: loaded library, resolved function-pointer
/// table, and the initialised NvAPI runtime. Owned once by
/// [`super::GpuCollector`] when at least one NVIDIA device is
/// detected; constructed by [`discover`] and dropped when the
/// collector drops. The `Drop` impl calls `NvAPI_Unload` exactly
/// once per successful `NvAPI_Initialize`.
pub(super) struct NvApiSession {
    nvapi: NvApiFunctions,
}

impl NvApiSession {
    /// Load `nvapi64.dll`, resolve the function table, and call
    /// `NvAPI_Initialize`. Returns `None` if the DLL is absent or
    /// initialise fails.
    fn load() -> Option<Self> {
        let nvapi = NvApiFunctions::load()?;

        // SAFETY: nvapi.initialize was resolved from nvapi64.dll
        // and matches the NvAPI_Initialize signature.
        let ret = unsafe { (nvapi.initialize)() };
        if ret != NVAPI_OK {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvapi,
                code = %crate::log::Hex(ret),
                "NvAPI_Initialize failed",
            );
            return None;
        }
        Some(Self { nvapi })
    }
}

impl Drop for NvApiSession {
    fn drop(&mut self) {
        // SAFETY: nvapi.unload was resolved from nvapi64.dll alongside
        // initialize. We call it exactly once per successful
        // `Initialize` (the session is constructed only when
        // initialize returns NVAPI_OK).
        unsafe {
            let _ = (self.nvapi.unload)();
        }
    }
}

/// Slim per-device NVIDIA state. One per detected GPU; held inside
/// [`super::GpuCollector`]'s `Vec<DeviceEntry>`. Carries only the
/// fields that genuinely vary per device — the function-pointer
/// table, library handle, and `NvAPI_Initialize` refcount live in
/// the singleton [`NvApiSession`] inside the parent
/// [`super::GpuCollector`].
pub(super) struct DeviceState {
    pub(super) handle: SendNvHandle,
    /// Matching NVML device handle, or `None` when NVML is unavailable
    /// or reported fewer devices than NvAPI. The handle stays valid
    /// for the collector's lifetime (it dies with [`NvmlSession`]).
    pub(super) nvml: Option<NvmlDevice>,
    /// Default power limit (TDP) in milliwatts, from NVML during
    /// discovery. Used to convert NvAPI Per-Cent-Mille power
    /// readings into milliwatts.
    pub(super) tdp_mw: u64,
}

/// One collection cycle for a single NVIDIA device.
///
/// Metrics are read from NVML first and from NvAPI second. The split
/// is not a preference: current NVIDIA drivers (570+) answer
/// `NVAPI_NOT_SUPPORTED` for the legacy NvAPI queries rtop
/// originally used for utilization, temperature, power, and clocks,
/// so on a modern driver the NvAPI leg is effectively dead code that
/// still serves older ones. A metric is only reported as degraded
/// when neither leg produced a reading.
///
/// Reads vendor function pointers from `session`, the per-device
/// handles and TDP from `dev`, mutates the rendered `info` in place,
/// and downgrades `status` on partial failures.
pub(super) fn collect(
    session: &NvApiSession,
    nvml: Option<&NvmlSession>,
    dev: &DeviceState,
    info: &mut crate::domain::gpu::GpuInfo,
    status: &mut CollectStatus,
) {
    let nvapi = &session.nvapi;
    let device = dev.handle.0;
    // Pair the session with this device's NVML handle once; `None`
    // when NVML is unavailable or has no handle for this adapter.
    let nvml = nvml.zip(dev.nvml);

    // ---- Utilization --------------------------------------------------
    let mut recorded = false;
    if let Some((nvml, handle)) = nvml {
        let mut util = NvmlUtilization::default();
        // SAFETY: handle is a live NVML device handle owned by the session.
        let ret = unsafe { (nvml.nvml.get_utilization_rates)(handle, &mut util) };
        if ret == NVML_SUCCESS {
            push_history(&mut info.gpu_percent.utilization, clamp_percent(util.gpu));
            recorded = true;
        } else {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvml,
                code = %crate::log::Hex(ret),
                "nvmlDeviceGetUtilizationRates failed",
            );
        }
    }
    if !recorded {
        let mut pstates = NvDynamicPstatesInfoEx {
            version: PSTATES_INFO_VER,
            flags: 0,
            utilizations: [NvUtilDomain::default(); 8],
        };
        // SAFETY: device is a valid handle obtained during discovery.
        let ret = unsafe { (nvapi.get_dynamic_pstates_info_ex)(device, &mut pstates) };
        if ret == NVAPI_OK {
            let gpu_util = &pstates.utilizations[0];
            if gpu_util.flags & 1 != 0 {
                push_history(
                    &mut info.gpu_percent.utilization,
                    clamp_percent(gpu_util.percentage),
                );
            }
        } else {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvapi,
                code = %crate::log::Hex(ret),
                "NvAPI utilization query failed",
            );
            status.downgrade(CollectStatus::Degraded("gpu utilization"));
        }
    }

    // ---- Temperature --------------------------------------------------
    let mut recorded = false;
    if let Some((nvml, handle)) = nvml {
        let mut temp: u32 = 0;
        // SAFETY: handle is a live NVML device handle owned by the session.
        let ret = unsafe { (nvml.nvml.get_temperature)(handle, NVML_TEMPERATURE_GPU, &mut temp) };
        if ret == NVML_SUCCESS {
            push_history(&mut info.temp, i64::from(temp));
            recorded = true;
        } else {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvml,
                code = %crate::log::Hex(ret),
                "nvmlDeviceGetTemperature failed",
            );
        }
    }
    if !recorded {
        // NvAPI fallback — request all sensors, pick the GPU target.
        let mut thermal = NvThermalSettings {
            version: THERMAL_SETTINGS_VER,
            count: 0,
            sensors: [NvThermalSensor::default(); 3],
        };
        // SAFETY: device is valid; NVAPI_THERMAL_TARGET_ALL requests all sensors.
        let ret =
            unsafe { (nvapi.get_thermal_settings)(device, NVAPI_THERMAL_TARGET_ALL, &mut thermal) };
        if ret == NVAPI_OK {
            let temp = thermal.sensors[..thermal.count as usize]
                .iter()
                .find(|s| s.target == NVAPI_THERMAL_TARGET_GPU)
                .or_else(|| thermal.sensors.first())
                .map(|s| s.current_temp as i64)
                .unwrap_or(0);
            push_history(&mut info.temp, temp);
        } else {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvapi,
                code = %crate::log::Hex(ret),
                "NvAPI temperature query failed",
            );
            status.downgrade(CollectStatus::Degraded("gpu temperature"));
        }
    }

    // ---- Memory (VRAM) ------------------------------------------------
    // NvAPI's memory query is still served by current drivers, so it
    // stays the single source for VRAM.
    let mut mem = NvMemoryInfo {
        version: MEMORY_INFO_VER,
        dedicated_video_memory_kb: 0,
        avail_dedicated_video_memory_kb: 0,
        system_video_memory_kb: 0,
        shared_system_memory_kb: 0,
        cur_avail_dedicated_video_memory_kb: 0,
        dedicated_video_memory_evictions_size_kb: 0,
        dedicated_video_memory_eviction_count: 0,
    };
    // SAFETY: device is valid; mem is a valid versioned struct.
    let ret = unsafe { (nvapi.get_memory_info)(device, &mut mem) };
    if ret == NVAPI_OK {
        let total = mem.dedicated_video_memory_kb as u64 * 1024;
        let avail = mem.cur_avail_dedicated_video_memory_kb as u64 * 1024;
        let used = total.saturating_sub(avail);
        info.mem_total = total;
        info.mem_used = used;
        let vram_pct = crate::collect::counters::percent_u64(used, total).min(100);
        push_history(&mut info.gpu_percent.vram, vram_pct);
        push_history(&mut info.mem_utilization_percent, vram_pct);
    } else {
        tracing::debug!(
            subsystem = %crate::log::Subsystem::GpuNvapi,
            code = %crate::log::Hex(ret),
            "NvAPI memory query failed",
        );
        status.downgrade(CollectStatus::Degraded("nvapi memory"));
    }

    // ---- Power --------------------------------------------------------
    let mut recorded = false;
    if let Some((nvml, handle)) = nvml {
        let mut milliwatts: u32 = 0;
        // SAFETY: handle is a live NVML device handle owned by the session.
        let ret = unsafe { (nvml.nvml.get_power_usage)(handle, &mut milliwatts) };
        if ret == NVML_SUCCESS {
            let power_mw = u64::from(milliwatts);
            info.pwr_usage = power_mw as i64;
            let pwr_pct = power_percent(power_mw, info.pwr_max_usage as u64);
            push_history(&mut info.gpu_percent.power, pwr_pct);
            recorded = true;
        } else {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvml,
                code = %crate::log::Hex(ret),
                "nvmlDeviceGetPowerUsage failed",
            );
        }
    }
    if !recorded && dev.tdp_mw > 0 {
        // NvAPI fallback — pick the Board domain entry (PCM) and
        // convert it to mW.
        let mut topo = NvPowerTopo {
            version: POWER_TOPO_VER,
            count: 0,
            entries: [NvPowerTopoEntry::default(); 4],
        };
        // SAFETY: device is valid; topo is a valid versioned struct.
        let ret = unsafe { (nvapi.client_power_topo_get_status)(device, &mut topo) };
        if ret == NVAPI_OK && topo.count > 0 {
            let n = (topo.count as usize).min(topo.entries.len());
            let board_pcm = topo.entries[..n]
                .iter()
                .find(|e| e.domain == POWER_DOMAIN_BOARD)
                .or_else(|| topo.entries.first())
                .map(|e| e.power_pcm as u64)
                .unwrap_or(0);
            let power_mw = pcm_to_mw(board_pcm, dev.tdp_mw);
            info.pwr_usage = power_mw as i64;
            let pwr_pct = power_percent(power_mw, info.pwr_max_usage as u64);
            push_history(&mut info.gpu_percent.power, pwr_pct);
        } else {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvapi,
                code = %crate::log::Hex(ret),
                "NvAPI power topology query failed",
            );
            status.downgrade(CollectStatus::Degraded("gpu power"));
        }
    }

    // ---- Graphics clock -----------------------------------------------
    let mut recorded = false;
    if let Some((nvml, handle)) = nvml {
        let mut megahertz: u32 = 0;
        // SAFETY: handle is a live NVML device handle owned by the session.
        let ret =
            unsafe { (nvml.nvml.get_clock_info)(handle, NVML_CLOCK_GRAPHICS, &mut megahertz) };
        if ret == NVML_SUCCESS {
            info.gpu_clock_speed = megahertz;
            recorded = true;
        } else {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvml,
                code = %crate::log::Hex(ret),
                "nvmlDeviceGetClockInfo failed",
            );
        }
    }
    if !recorded {
        let mut clocks = NvClockFrequencies {
            version: CLOCK_FREQUENCIES_VER,
            clock_type: CLOCK_TYPE_CURRENT,
            domains: [NvClockDomain::default(); 32],
        };
        // SAFETY: device is valid; clocks is a valid versioned struct.
        let ret = unsafe { (nvapi.get_all_clock_frequencies)(device, &mut clocks) };
        if ret == NVAPI_OK {
            let gfx = &clocks.domains[CLOCK_DOMAIN_GRAPHICS];
            if gfx.flags & 1 != 0 {
                info.gpu_clock_speed = gfx.frequency_khz / 1000;
            }
        } else {
            tracing::debug!(
                subsystem = %crate::log::Subsystem::GpuNvapi,
                code = %crate::log::Hex(ret),
                "NvAPI clock query failed",
            );
            status.downgrade(CollectStatus::Degraded("gpu clock"));
        }
    }
}

/// Discovery result for the NVIDIA vendor.
///
/// Returned by [`discover`] when at least one device was detected;
/// owned by [`super::GpuCollector`] which holds the session for the
/// lifetime of every device entry derived from this bundle.
pub(super) struct NvApiBundle {
    pub(super) session: NvApiSession,
    /// Live NVML session, or `None` when NVML could not be opened.
    /// Declared after `session` so it outlives the per-device NVML
    /// handles held in `entries`.
    pub(super) nvml: Option<NvmlSession>,
    pub(super) entries: Vec<(DeviceState, crate::domain::gpu::GpuInfo)>,
}

/// Discover every NVIDIA GPU. Returns `None` when NvAPI is
/// unavailable or no GPUs are detected (vendor session drops on the
/// spot — no library / init resources outlive an empty discovery).
pub(super) fn discover() -> Option<NvApiBundle> {
    let session = NvApiSession::load()?;

    let mut handles = [std::ptr::null_mut::<c_void>(); NVAPI_MAX_PHYSICAL_GPUS];
    let mut count: u32 = 0;
    // SAFETY: handles is a valid array of NVAPI_MAX_PHYSICAL_GPUS
    // pointers; count is a valid pointer to a stack-allocated u32.
    let ret = unsafe { (session.nvapi.enum_physical_gpus)(handles.as_mut_ptr(), &mut count) };
    if ret != NVAPI_OK {
        tracing::warn!(
            subsystem = %crate::log::Subsystem::GpuNvapi,
            code = %crate::log::Hex(ret),
            "NvAPI_EnumPhysicalGPUs failed",
        );
        return None;
    }
    let count = (count as usize).min(NVAPI_MAX_PHYSICAL_GPUS);
    if count == 0 {
        return None;
    }
    let devices: Vec<NvPhysicalGpuHandle> = handles[..count].to_vec();

    // Open an NVML session: it carries the per-device handles used by
    // every collect cycle plus the static metadata (default TDP for
    // the PCM→mW conversion and the stable UUID for cross-run
    // identity). The device list is padded to `count` entries with
    // `None` so the zip always lines up; slots NVML did not report
    // fall back to TDP 0 + empty UUID and NvAPI-only metrics.
    let nvml = NvmlSession::load();
    let nvml_devices: Vec<Option<NvmlDeviceEntry>> = nvml
        .as_ref()
        .map(|session| session.devices(count))
        .unwrap_or_default();

    tracing::info!(
        subsystem = %crate::log::Subsystem::GpuNvapi,
        devices = count,
        nvml = nvml.is_some(),
        "vendor initialized",
    );

    let entries: Vec<(DeviceState, crate::domain::gpu::GpuInfo)> = devices
        .into_iter()
        .zip(nvml_devices)
        .enumerate()
        .map(|(idx, (device, meta))| {
            let (nvml_handle, tdp_mw, uuid) = match meta {
                Some(meta) => (Some(meta.handle), u64::from(meta.tdp_mw), meta.uuid),
                None => (None, 0, String::new()),
            };
            let stable_id = if uuid.is_empty() {
                tracing::debug!(
                    subsystem = %crate::log::Subsystem::GpuNvapi,
                    device = idx,
                    "NVML UUID unavailable; using vendor-relative adapter index fallback",
                );
                format!("NVIDIA:adapter:{idx}")
            } else {
                format!("NVIDIA:{uuid}")
            };
            let info = build_initial_info(
                &session.nvapi,
                nvml.as_ref().zip(nvml_handle),
                device,
                tdp_mw,
                stable_id,
            );
            (
                DeviceState {
                    handle: SendNvHandle(device),
                    nvml: nvml_handle,
                    tdp_mw,
                },
                info,
            )
        })
        .collect();

    Some(NvApiBundle {
        session,
        nvml,
        entries,
    })
}

/// Pre-fill the per-device `GpuInfo` with static fields (stable
/// id, name, max-power reference, max boost clock) that the
/// collector renders but never re-queries on each cycle. Mirrors
/// AMD's [`build_initial_info`](super::amd) helper for parity.
fn build_initial_info(
    nvapi: &NvApiFunctions,
    nvml: Option<(&NvmlSession, NvmlDevice)>,
    device: NvPhysicalGpuHandle,
    tdp_mw: u64,
    stable_id: String,
) -> crate::domain::gpu::GpuInfo {
    let mut info = crate::domain::gpu::GpuInfo {
        stable_id,
        ..crate::domain::gpu::GpuInfo::default()
    };

    let mut name_buf = [0u8; NVAPI_SHORT_STRING_MAX];
    // SAFETY: device is a valid handle; name_buf is 64 bytes
    // matching NvAPI_ShortString.
    let ret = unsafe { (nvapi.gpu_get_full_name)(device, name_buf.as_mut_ptr() as *mut c_char) };
    if ret == NVAPI_OK {
        info.name = crate::collect::win::string_from_c_buf(&name_buf);
    }

    // Power limit — use TDP from NVML, scaled by the NvAPI power
    // policy max percentage (PCM) when the user has raised the
    // limit above default.
    if tdp_mw > 0 {
        info.pwr_max_usage = tdp_mw as i64;

        let mut power_info = NvPowerInfo {
            version: POWER_INFO_VER,
            valid: 0,
            count: 0,
            padding: [0; 2],
            entries: [NvPowerInfoEntry::default(); 4],
        };
        // SAFETY: device is valid; power_info is a valid versioned struct.
        let ret = unsafe { (nvapi.client_power_policies_get_info)(device, &mut power_info) };
        if ret == NVAPI_OK && power_info.count > 0 {
            let max_pcm = power_info.entries[0].max_power_pcm as u64;
            if max_pcm > PCM_100_PERCENT {
                info.pwr_max_usage = pcm_to_mw(max_pcm, tdp_mw) as i64;
            }
        }
    }

    let mut clocks = NvClockFrequencies {
        version: CLOCK_FREQUENCIES_VER,
        clock_type: CLOCK_TYPE_BOOST,
        domains: [NvClockDomain::default(); 32],
    };
    // SAFETY: device is valid; clocks is a valid versioned struct.
    let ret = unsafe { (nvapi.get_all_clock_frequencies)(device, &mut clocks) };
    if ret == NVAPI_OK {
        let gfx = &clocks.domains[CLOCK_DOMAIN_GRAPHICS];
        if gfx.flags & 1 != 0 {
            info.gpu_max_clock_speed = gfx.frequency_khz / 1000;
        }
    } else if let Some((nvml, handle)) = nvml {
        // NvAPI's boost-clock query is one of the entry points current
        // drivers refuse; NVML still serves the max clock.
        if let Some(get_max_clock_info) = nvml.nvml.get_max_clock_info {
            let mut megahertz: u32 = 0;
            // SAFETY: handle is a live NVML device handle owned by the session.
            let nvml_ret =
                unsafe { get_max_clock_info(handle, NVML_CLOCK_GRAPHICS, &mut megahertz) };
            if nvml_ret == NVML_SUCCESS {
                info.gpu_max_clock_speed = megahertz;
            } else {
                tracing::debug!(
                    subsystem = %crate::log::Subsystem::GpuNvml,
                    code = %crate::log::Hex(nvml_ret),
                    "nvmlDeviceGetMaxClockInfo failed",
                );
            }
        }
    }

    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_sizes_match_nvapi_layout() {
        assert_eq!(std::mem::size_of::<NvUtilDomain>(), 8);
        assert_eq!(std::mem::size_of::<NvDynamicPstatesInfoEx>(), 72);
        assert_eq!(std::mem::size_of::<NvThermalSensor>(), 20);
        assert_eq!(std::mem::size_of::<NvThermalSettings>(), 68);
        assert_eq!(std::mem::size_of::<NvMemoryInfo>(), 32);
        assert_eq!(std::mem::size_of::<NvClockDomain>(), 8);
        assert_eq!(std::mem::size_of::<NvClockFrequencies>(), 264);
        assert_eq!(std::mem::size_of::<NvPowerTopoEntry>(), 16);
        assert_eq!(std::mem::size_of::<NvPowerTopo>(), 72);
        assert_eq!(std::mem::size_of::<NvPowerInfoEntry>(), 44);
        assert_eq!(std::mem::size_of::<NvPowerInfo>(), 184);
        assert_eq!(std::mem::size_of::<NvmlUtilization>(), 8);
    }

    #[test]
    fn version_constants_encode_correctly() {
        assert_eq!(PSTATES_INFO_VER, 72 | (1 << 16));
        assert_eq!(THERMAL_SETTINGS_VER, 68 | (2 << 16));
        assert_eq!(MEMORY_INFO_VER, 32 | (3 << 16));
        assert_eq!(CLOCK_FREQUENCIES_VER, 264 | (3 << 16));
        assert_eq!(POWER_TOPO_VER, 72 | (1 << 16));
        assert_eq!(POWER_INFO_VER, 184 | (1 << 16));
    }
}
