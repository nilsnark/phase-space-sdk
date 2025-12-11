# Phase Space Context SDK API

This document summarizes the external SDK surface intended for third-party plugins. It mirrors the public modules in `src/crates/context_sdk` and describes the main data transfer objects (DTOs), traits, and utilities exposed to plugin and scripting authors.

## Crate layout

* `math`: Deterministic math DTOs (`Vec3`, `Mat3`, `Mat4`, `Quaternion`).
* `handles`: Stable identifiers for entities, dimensions, and ticks.
* `view`: Read-only world views surfaced to plugins.
* `world`: DTOs describing transforms, velocity, physics components, and engine profiles.
* `determinism`: Deterministic RNG helpers and stateful script RNGs.
* `sensors`: DTOs and helpers for defining sensors, requests, observations, and running sensor simulations.
* `context`: Deterministic identifiers, state blobs, actors, brains, and intent envelopes.
* `intent`: Re-export of intent envelopes used by plugins and systems.
* `plugin`: Narrow plugin trait and context events.
* `abi`: FFI-friendly ABI structs shared with WASM plugins.
* `guest`: WASM guest wrappers over host callbacks (only compiled for `wasm32`).
* `script`: Script runtime helpers (mailboxes, brain references, runtime traits, and self views).
* `runtime`: Scheduler/runtime-facing abstractions used by SDK-driven simulations.

## Math DTOs (`math`)

### `Vec3`
* Fields: `x`, `y`, `z` (all `f64`).
* Constructors and helpers: `zero()`, `new(x, y, z)`, `length_squared()`, `length()`, `normalized()`, `dot()`, `cross()`, `is_finite()`, `norm()`.
* Arithmetic: `Add`, `Sub`, scalar `Mul<f64>`, scalar `Div<f64>`.

### `Mat3`
* Row-major 3×3 matrix stored as `[Vec3; 3]`.
* Constructors: `identity()`, `from_rows(row0, row1, row2)`.
* Operations: `multiply(rhs)`, `mul_vec3(v)`, `transpose()`, `determinant()`, `is_finite()`.
* Operator overloads: matrix × vector (`Mul<Vec3>`), matrix × matrix (`Mul<Mat3>`).

### `Mat4`
* Row-major 4×4 matrix stored as `[[f64; 4]; 4]`.
* Constructors: `identity()`, `from_rows(rows)`.
* Operations: `multiply(rhs)`, `mul_vec3(v)` (homogeneous w=1), `transpose()`, `is_finite()`.
* Operator overloads: matrix × vector (`Mul<Vec3>`), matrix × matrix (`Mul<Mat4>`).

### `Quaternion`
* Fields: `w`, `x`, `y`, `z` (unit quaternion expected for rotation).
* Constructors and helpers: `identity()`, `new(w, x, y, z)`, `norm_squared()`, `norm()`, `is_normalized()`, `normalized()`.
* Operations: `rotate_vec3(v)`, `multiply(rhs)`, `from_angular_velocity(omega)`.
* Arithmetic: `Add`, scalar `Mul<f64>`.

## Handles and views (`handles`, `view`)

* `EntityId(u64)`, `DimensionId(u32)`, and tick alias `Tick = u64` serve as deterministic identifiers.
* `WorldViewSnapshot` captures `world_seed`, `dimension`, `engine_tick`, and `dimension_tick` and implements the `WorldView` trait.
* `WorldView` exposes read-only accessors for deterministic metadata; plugins receive `&dyn WorldView` in callbacks.

## World DTOs (`world`)

* Physics profile: `CORE_PHYSICS_PROFILE_INTERSTELLAR` constant.
* Component DTOs: `TransformDto { position_m: Vec3 }`, `VelocityDto { linear_velocity_m_per_s: Vec3 }`, `MassDto { mass }`, `EngineDto { max_thrust_n, max_vector_angle_rad, specific_impulse_s, throttle, active, vector_angle_rad }`, `FuelTankDto { fuel_type_id, max_capacity_kg, current_mass_kg, consumption_enabled }`, `ManeuverNodeDto { start_time_s, duration_s, thrust_x, thrust_y, is_delta_v, accumulated_burn_time_s }`.
* Dimension helpers: `DimensionTag` enum with `dimension_type_id()` mapping common dimension types; `FrameRef` for frame identifiers.

## Deterministic RNG helpers (`determinism`)

* Seed and domain: `WorldSeed = u64`, `RngDomain` enum (Engine, Sensors, Scripts, Procgen, PhysicsNoise, Ui) with internal salt for deterministic streams.
* Stateless functions: `engine_rand_u64(seed, domain, a, b, c)` and `engine_rand_f64(seed, domain, a, b, c)` generate deterministic values without shared state.
* Stateful script RNG: `ScriptRngState` wraps `Pcg64Mcg` with `from_seed`, `from_raw`, `as_raw`, `as_raw_mut`, `into_raw`, `clone_raw`, `next_u64`, `next_f64`.

## Sensors (`sensors`)

* Identifiers: `SensorId(u64)`, `BandId(u32)`, `ObservationKind(u32)`.
* Resolution DTO: `Resolution { range_resolution, angular_resolution }` with `new` constructor.
* Core DTOs:
  * `SensorDef` describes static sensor parameters (id, band, FOV, range, aperture, resolution, integration ticks, cooldown, payload noise scale).
  * `ScanRequest` encodes the current scan command (target direction, overrides for integration ticks, FOV, range, aperture).
  * `SensorState` tracks cooldown and integration progress with helpers `idle()`, `integrating`, and `cooling_down`.
  * `Target { entity, range, line_of_sight }` describes an observed target.
  * `ObservationMeta` captures measurement context (band, integration, aperture, FOV, max range, resolution, scan direction).
  * `Observation` pairs sensor/time metadata with payload, SNR, and optional `target_hint`.
* `SensorWorld` trait abstracts the minimal world surface needed by sensor simulations (access to seeds, transforms, actors/brains, sensor defs/states/requests, and per-entity lists).
* `SensorRunner<F>` executes a sensor pass: `new(compute_observation)` constructs with a custom callback returning an `IdealObservation`; `run` iterates over entities, applies FOV/range checks, injects deterministic noise, buffers observations per brain, and updates `SensorState`. Helpers include `drain_observations` and `observations` for buffered results.

## Context and intents (`context`, `intent`)

* Identifiers and payloads: `BrainId(u64)`, `DeterministicContextBlob(Vec<u8>)` with `new`, `as_slice`, and mutable `BrainState` wrapper.
* Components: `Actor { brain: BrainId }`, `BrainDto { brain: BrainId, profile_id: u32 }` describing the brain metadata associated with an entity.
* Intent envelope: `IntentEnvelope<P>` carries `issued_by`, `issued_at`, `applies_at`, `payload`, and optional `dimension`. Methods `target_dimension(default)` and `for_dimension(dimension)` help route intents.

## Plugin surface (`plugin`)

* `ContextPlugin` trait defines the narrow plugin interface:
  * Associated `Intent` type (`Clone + Send + 'static`).
  * Lifecycle hooks: `on_tick(&mut self, &dyn WorldView) -> Vec<IntentEnvelope<Intent>>` and optional `on_event(ContextEvent)`.
  * Downcast helpers `as_any`/`as_any_mut`.
* `ContextEvent` variants include `EngineTick { engine_tick, dimension }` and `Custom { id, dimension }`.

## ABI for WASM plugins (`abi`)

* Status codes: `HostCallStatus` (`Ok`, `MissingWorld`, `MissingMemory`, `InvalidEntity`, `IntentRejected`) with `from_i32` conversion.
* FFI DTOs:
  * `PluginEntity { id, dimension, reserved }` and `PluginVec3 { x, y, z }`.
  * `PluginEntitySnapshot` captures entity, transform/velocity flags, positions, velocities, and throttle; `empty(entity)` zero-fills a snapshot.
  * `PluginIntentKind` enum (`None`, `SetThrottle`) with `from_i32` conversion.
  * `PluginIntentEnvelope` contains target and issuer identifiers, timing, intent kind, and scalar payload.
  * `PluginTickInfo` carries world seed, engine/dimension ticks, and dimension metadata.

## WASM guest helpers (`guest`)

* Compiled only for `wasm32` targets; wraps host callbacks `log`, `get_entity_snapshot`, `find_first_entity_with_engine`, and `submit_intent`.
* Safe wrappers emit `HostCallStatus` and typed DTOs: `log`, `entity_snapshot`, `first_entity_with_engine`, `submit_intent`.
* Convenience helpers: `PluginEntity::new`/`root`, `PluginEntitySnapshot::empty`, `PluginIntentEnvelope::new`, and `PluginTickInfo::from_context`.

## Script helpers (`script`)

* Identifiers: `ScriptId = u64`, `BrainRef { brain_id, entity, dimension }`.
* Mailbox DTO: `Mailbox { inbox, outbox }` with `new` constructor.
* Errors/results: `ScriptError` (`InvalidParameter`, `MissingScript`, `BudgetExceeded`), `ScriptResult<T>` alias.
* Traits: `ScriptContextBuilder` builds per-brain contexts from a `RuntimeWorld`, brain ref, optional `BrainState`, and optional `Mailbox`; `ScriptRuntime` loads/initializes/ticks scripts with a user-provided context type.
* Self view: `SelfView` snapshots the owning entity (actor, transform, velocity, mass, engine, fuel tank) with `new(entity, dimension)` initializer.

## Runtime and scheduling (`runtime`)

The `runtime` module provides deterministic scheduling utilities for simulations that do not depend on the engine scheduler.

* Tick phases: `TickPhase` enum (`Sensors`, `Scripts`, `IntentCommit`, `Physics`, `Custom(u8)`) with `as_index()` for array indexing.
* Phase ordering: `PhaseOrder` deduplicates and stores an ordered list of phases; defaults to Physics → Sensors → Scripts → IntentCommit.
* Dimension profiles: `DimensionProfile` holds a `PhaseOrder` and registered `SystemId`s per phase.
* Clocks: `SimDuration` (microsecond-based), `EngineClock` (global tick + elapsed micros), and `DimensionClock` (per-dimension time scale, accumulation, pending tick tracking).
* Scheduling metadata: `DimensionSchedule` (clock, priority, tick budget, fixed timestep override, pause flag, effective tick rate override, profile), `ResolvedTick`, `ResolvedPhase`, and `TickSchedule` (per-engine-step resolution of ticks/phases).
* Scheduler types: `SimulationScheduler` advances dimension clocks and resolves work for each engine step; `Scheduler` groups systems by phase and runs them with timing info; `SimulationLoop` wires the two together, running phases for each scheduled tick and returning the `TickSchedule`.
* Runtime world surface: `RuntimeWorld` trait exposes minimal getters/setters for brains, sensors, transforms, velocities, mass, mailboxes, script RNG, and intent application. Default methods are no-ops, allowing engines to provide concrete implementations.
* Systems: `System` trait defines `run(&mut self, &mut dyn RuntimeWorld, dt, &mut SystemRunContext)`, where `SystemRunContext` captures dimension and tick metadata.

## Testing

The SDK includes basic determinism tests validating stateless RNG behavior and `ScriptRngState` floating-point ranges.
