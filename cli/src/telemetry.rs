use std::{fs::File, path::Path};

use pprof::{ProfilerGuard, ProfilerGuardBuilder};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Initialise tracing output and start a CPU profiler.
pub fn init_telemetry() -> Option<ProfilerGuard<'static>> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = fmt::layer().with_target(false);
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libpthread", "libgcc", "libm"])
        .build()
        .ok()
}

/// Persist the collected CPU profile if profiling was active.
pub fn write_profile(guard: ProfilerGuard<'_>, output_path: impl AsRef<Path>) {
    if let Ok(report) = guard.report().build() {
        if let Ok(mut file) = File::create(output_path) {
            let _ = report.flamegraph(&mut file);
        }
    }
}
