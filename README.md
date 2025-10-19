# SysPike

Prototype Rust workspace showcasing an event-driven reasoning pipeline with a diffusion-based stabilisation loop. The project centres on a single binary that can train, evaluate, and run inference locally without external services.

## Documentation

The library is documented in-source via Rustdoc comments. Generate the HTML documentation with `cargo doc --open`.

An architectural walk-through covering crate responsibilities and data flow lives in
[`docs/architecture_overview.md`](docs/architecture_overview.md).

## Testing

Run the verification suite with `cargo test`.
