# Phase Space Context SDK

`phase_space_context_sdk` is the external-facing, sanitized SDK for third-party
plugins. It exposes deterministic identifiers, read-only world views, and a
minimal plugin trait without surfacing the engine's `World`/`Scheduler`
internals. First-party code and engine tests should depend on
`phase_space_core::context_sdk` directly for the full surface area.

Plugins can implement `plugin::ContextPlugin` and, when building inside the
engine workspace, wrap it with `plugin::EngineAdapter` by enabling the
`engine_internal` cargo feature. See
[`docs/engine/context-plugin-sdk.md`](../../docs/engine/context-plugin-sdk.md)
for authoring guidance, ABI details, and examples.

The SDK ships simple math DTOs (`Vec3`, `Quaternion`, `Mat3`, `Mat4`) under
`phase_space_context_sdk::math` so plugin authors do not need to depend on the
engine's `phase_space_physics` crate. Engine-facing code can convert between
SDK DTOs and physics types via the internal `phase_space_context_sdk_internal`
crate when bridging to integrators or components.
