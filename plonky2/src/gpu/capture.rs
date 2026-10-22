//! GPU debugger captures of one proof phase, for Metal's shader profiler.
//!
//! `PLONKY2_GPU_CAPTURE=<file.gputrace>` captures the wires commitment, and
//! `PLONKY2_GPU_CAPTURE_QUOTIENT=<file.gputrace>` the quotient evaluation,
//! with `PLONKY2_GPU_CAPTURE_INDEX` selecting the zero-based invocation of each
//! phase (default 1, the second invocation). The file must not exist yet.
//! The process needs `METAL_CAPTURE_ENABLED=1` in its
//! environment. Other platforms and unset variables make this a no-op.

use std::sync::atomic::{AtomicUsize, Ordering};

static WIRES_CALLS: AtomicUsize = AtomicUsize::new(0);
static QUOTIENT_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Stops the capture it started when dropped.
#[derive(Debug)]
pub struct Capture {
    active: bool,
}

impl Capture {
    /// Starts a capture if this is the selected wires commitment.
    pub fn wires() -> Capture {
        Self::phase("PLONKY2_GPU_CAPTURE", "wires commitment", &WIRES_CALLS)
    }

    /// Starts a capture if this is the selected quotient evaluation
    /// (`PLONKY2_GPU_CAPTURE_QUOTIENT=<file.gputrace>`).
    pub fn quotient() -> Capture {
        Self::phase(
            "PLONKY2_GPU_CAPTURE_QUOTIENT",
            "quotient evaluation",
            &QUOTIENT_CALLS,
        )
    }

    fn phase(variable: &str, what: &str, calls: &AtomicUsize) -> Capture {
        let call = calls.fetch_add(1, Ordering::Relaxed);
        let Some(path) = crate::util::switches::text(variable) else {
            return Capture { active: false };
        };
        let index =
            crate::util::switches::number::<usize>("PLONKY2_GPU_CAPTURE_INDEX").unwrap_or(1);
        if call != index {
            return Capture { active: false };
        }
        let active = start(&path);
        if active {
            log::info!("GPU capture of {what} {call} to {path}");
        }
        Capture { active }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if self.active {
            stop();
            log::info!("GPU capture stopped");
        }
    }
}

#[cfg(target_os = "macos")]
fn start(path: &str) -> bool {
    use std::ops::Deref;

    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

    let Some(ctx) = super::context() else {
        log::warn!("GPU capture requested without a GPU context");
        return false;
    };
    // SAFETY: the guard is dropped before this function returns and the raw
    // device is only read.
    let Some(hal) = (unsafe { ctx.device.as_hal::<wgpu::hal::api::Metal>() }) else {
        log::warn!("GPU capture requested on a non-Metal device");
        return false;
    };
    let device = hal.raw_device().deref();
    // SAFETY: Metal's shared capture manager is a process singleton.
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
        log::warn!("GPU capture to a file is not supported (is METAL_CAPTURE_ENABLED=1 set?)");
        return false;
    }
    let descriptor = MTLCaptureDescriptor::new();
    // SAFETY: a device is a valid capture object.
    unsafe { descriptor.setCaptureObject(Some(device.as_ref())) };
    descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
    let url = NSURL::fileURLWithPath(&NSString::from_str(path));
    descriptor.setOutputURL(Some(&url));
    match manager.startCaptureWithDescriptor_error(&descriptor) {
        Ok(()) => true,
        Err(error) => {
            log::warn!("GPU capture failed to start: {error}");
            false
        }
    }
}

#[cfg(target_os = "macos")]
fn stop() {
    // SAFETY: Metal's shared capture manager is a process singleton.
    unsafe { objc2_metal::MTLCaptureManager::sharedCaptureManager() }.stopCapture();
}

#[cfg(not(target_os = "macos"))]
fn start(_path: &str) -> bool {
    log::warn!("GPU captures are only implemented for Metal");
    false
}

#[cfg(not(target_os = "macos"))]
fn stop() {}
