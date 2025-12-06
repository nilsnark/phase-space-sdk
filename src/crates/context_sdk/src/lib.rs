//! External-facing context SDK that avoids exposing engine internals.
//!
//! This crate is the narrow, sanitized surface intended for third-party
//! plugins. It provides deterministic identifiers, read-only world views, and
//! a small plugin trait that emits intent envelopes without giving access to
//! the engine's `World` or `Scheduler`. Engine and first-party code should use
//! `phase_space_core::context_sdk` directly for full control; plugins should
//! depend on this crate and wrap themselves with [`plugin::EngineAdapter`]
//! when exporting the entrypoint expected by the loader.

/// Deterministic math DTOs available to plugin authors without depending on
/// the full physics crate.
pub mod math {
    use serde::{Deserialize, Serialize};
    use std::ops::{Add, Div, Mul, Sub};

    /// Simple 3D vector with f64 precision.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Vec3 {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    impl Vec3 {
        /// Zero vector.
        pub fn zero() -> Self {
            Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        }

        /// Construct a vector from components.
        pub fn new(x: f64, y: f64, z: f64) -> Self {
            Self { x, y, z }
        }

        /// Squared length of the vector.
        pub fn length_squared(self) -> f64 {
            self.x * self.x + self.y * self.y + self.z * self.z
        }

        /// Length (magnitude) of the vector.
        pub fn length(self) -> f64 {
            self.length_squared().sqrt()
        }

        /// Return a normalized vector; returns zero if the length is effectively zero.
        pub fn normalized(self) -> Self {
            let len = self.length();
            if len <= f64::EPSILON {
                Self::zero()
            } else {
                self / len
            }
        }

        /// Dot product of two vectors.
        pub fn dot(self, other: Self) -> f64 {
            self.x * other.x + self.y * other.y + self.z * other.z
        }

        /// Cross product of two vectors.
        pub fn cross(self, other: Self) -> Self {
            Self {
                x: self.y * other.z - self.z * other.y,
                y: self.z * other.x - self.x * other.z,
                z: self.x * other.y - self.y * other.x,
            }
        }

        /// Check if all components are finite (not NaN or infinity).
        pub fn is_finite(self) -> bool {
            self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
        }

        /// Alias for length() to match design doc terminology.
        pub fn norm(self) -> f64 {
            self.length()
        }
    }

    impl Add for Vec3 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
            }
        }
    }

    impl Sub for Vec3 {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
                z: self.z - rhs.z,
            }
        }
    }

    impl Mul<f64> for Vec3 {
        type Output = Self;

        fn mul(self, rhs: f64) -> Self::Output {
            Self {
                x: self.x * rhs,
                y: self.y * rhs,
                z: self.z * rhs,
            }
        }
    }

    impl Div<f64> for Vec3 {
        type Output = Self;

        fn div(self, rhs: f64) -> Self::Output {
            Self {
                x: self.x / rhs,
                y: self.y / rhs,
                z: self.z / rhs,
            }
        }
    }

    /// 3x3 matrix with row-major storage.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Mat3 {
        pub rows: [Vec3; 3],
    }

    impl Mat3 {
        /// Identity matrix.
        pub fn identity() -> Self {
            Self {
                rows: [
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                ],
            }
        }

        /// Construct from row vectors.
        pub fn from_rows(row0: Vec3, row1: Vec3, row2: Vec3) -> Self {
            Self {
                rows: [row0, row1, row2],
            }
        }

        /// Matrix multiplication.
        pub fn multiply(self, rhs: Self) -> Self {
            let cols = rhs.transpose();
            Self::from_rows(
                Vec3::new(
                    self.rows[0].dot(cols.rows[0]),
                    self.rows[0].dot(cols.rows[1]),
                    self.rows[0].dot(cols.rows[2]),
                ),
                Vec3::new(
                    self.rows[1].dot(cols.rows[0]),
                    self.rows[1].dot(cols.rows[1]),
                    self.rows[1].dot(cols.rows[2]),
                ),
                Vec3::new(
                    self.rows[2].dot(cols.rows[0]),
                    self.rows[2].dot(cols.rows[1]),
                    self.rows[2].dot(cols.rows[2]),
                ),
            )
        }

        /// Multiply by a vector.
        pub fn mul_vec3(self, v: Vec3) -> Vec3 {
            Vec3::new(
                self.rows[0].dot(v),
                self.rows[1].dot(v),
                self.rows[2].dot(v),
            )
        }

        /// Transpose of the matrix.
        pub fn transpose(self) -> Self {
            Self::from_rows(
                Vec3::new(self.rows[0].x, self.rows[1].x, self.rows[2].x),
                Vec3::new(self.rows[0].y, self.rows[1].y, self.rows[2].y),
                Vec3::new(self.rows[0].z, self.rows[1].z, self.rows[2].z),
            )
        }

        /// Determinant of the matrix.
        pub fn determinant(self) -> f64 {
            let m = self.rows;
            m[0].x * (m[1].y * m[2].z - m[1].z * m[2].y)
                - m[0].y * (m[1].x * m[2].z - m[1].z * m[2].x)
                + m[0].z * (m[1].x * m[2].y - m[1].y * m[2].x)
        }

        /// Check if all entries are finite.
        pub fn is_finite(self) -> bool {
            self.rows.iter().all(|row| row.is_finite())
        }
    }

    impl Mul<Vec3> for Mat3 {
        type Output = Vec3;

        fn mul(self, rhs: Vec3) -> Self::Output {
            self.mul_vec3(rhs)
        }
    }

    impl Mul<Mat3> for Mat3 {
        type Output = Mat3;

        fn mul(self, rhs: Mat3) -> Self::Output {
            self.multiply(rhs)
        }
    }

    /// 4x4 homogeneous transform matrix with row-major storage.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Mat4 {
        pub rows: [[f64; 4]; 4],
    }

    impl Mat4 {
        /// Identity matrix.
        pub fn identity() -> Self {
            Self {
                rows: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        }

        /// Construct from rows.
        pub fn from_rows(rows: [[f64; 4]; 4]) -> Self {
            Self { rows }
        }

        /// Multiply two Mat4 values.
        pub fn multiply(self, rhs: Self) -> Self {
            let mut result = [[0.0; 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    result[r][c] = self.rows[r][0] * rhs.rows[0][c]
                        + self.rows[r][1] * rhs.rows[1][c]
                        + self.rows[r][2] * rhs.rows[2][c]
                        + self.rows[r][3] * rhs.rows[3][c];
                }
            }
            Self { rows: result }
        }

        /// Multiply by a Vec3 (assumes homogeneous w=1).
        pub fn mul_vec3(self, v: Vec3) -> Vec3 {
            let x = self.rows[0][0] * v.x
                + self.rows[0][1] * v.y
                + self.rows[0][2] * v.z
                + self.rows[0][3];
            let y = self.rows[1][0] * v.x
                + self.rows[1][1] * v.y
                + self.rows[1][2] * v.z
                + self.rows[1][3];
            let z = self.rows[2][0] * v.x
                + self.rows[2][1] * v.y
                + self.rows[2][2] * v.z
                + self.rows[2][3];
            let w = self.rows[3][0] * v.x
                + self.rows[3][1] * v.y
                + self.rows[3][2] * v.z
                + self.rows[3][3];

            if (w - 1.0).abs() > f64::EPSILON && w.abs() > f64::EPSILON {
                Vec3::new(x / w, y / w, z / w)
            } else {
                Vec3::new(x, y, z)
            }
        }

        /// Transpose of the matrix.
        pub fn transpose(self) -> Self {
            let mut result = [[0.0; 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    result[r][c] = self.rows[c][r];
                }
            }
            Self { rows: result }
        }

        /// Check if all entries are finite.
        pub fn is_finite(self) -> bool {
            self.rows.iter().flatten().all(|v| v.is_finite())
        }
    }

    impl Mul<Vec3> for Mat4 {
        type Output = Vec3;

        fn mul(self, rhs: Vec3) -> Self::Output {
            self.mul_vec3(rhs)
        }
    }

    impl Mul<Mat4> for Mat4 {
        type Output = Mat4;

        fn mul(self, rhs: Mat4) -> Self::Output {
            self.multiply(rhs)
        }
    }

    /// Unit quaternion for representing 3D rotations.
    /// Stored as w + xi + yj + zk where w is the scalar part.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Quaternion {
        pub w: f64,
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    impl Quaternion {
        /// Tolerance for quaternion normalization checks.
        const QUATERNION_TOLERANCE: f64 = 1e-6;

        /// Identity quaternion (no rotation).
        pub fn identity() -> Self {
            Self {
                w: 1.0,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        }

        /// Construct a quaternion from components (w, x, y, z).
        pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
            Self { w, x, y, z }
        }

        /// Compute the squared norm of the quaternion.
        pub fn norm_squared(self) -> f64 {
            self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
        }

        /// Compute the norm (magnitude) of the quaternion.
        pub fn norm(self) -> f64 {
            self.norm_squared().sqrt()
        }

        /// Check if the quaternion is normalized (unit quaternion).
        pub fn is_normalized(self) -> bool {
            (self.norm_squared() - 1.0).abs() < Self::QUATERNION_TOLERANCE
        }

        /// Normalize the quaternion to unit length.
        /// Returns identity quaternion for zero or near-zero input.
        pub fn normalized(self) -> Self {
            let n = self.norm();
            if n <= f64::EPSILON {
                #[cfg(debug_assertions)]
                {
                    if n > 0.0 {
                        eprintln!("Warning: normalizing near-zero quaternion (norm={})", n);
                    }
                }
                Self::identity()
            } else {
                Self {
                    w: self.w / n,
                    x: self.x / n,
                    y: self.y / n,
                    z: self.z / n,
                }
            }
        }

        /// Rotate a vector by this quaternion.
        /// Uses the formula: v' = q * v * q^-1
        pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
            let qv = Quaternion::new(0.0, v.x, v.y, v.z);
            let q_conj = Quaternion::new(self.w, -self.x, -self.y, -self.z);
            let result = self.multiply(qv).multiply(q_conj);
            Vec3::new(result.x, result.y, result.z)
        }

        /// Quaternion multiplication.
        pub fn multiply(self, rhs: Self) -> Self {
            Self {
                w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
                x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
                y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
                z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            }
        }

        /// Create a pure quaternion from an angular velocity vector for use in integration.
        /// This represents the angular velocity as a pure quaternion (0, ωx, ωy, ωz).
        pub fn from_angular_velocity(omega: Vec3) -> Self {
            Self {
                w: 0.0,
                x: omega.x,
                y: omega.y,
                z: omega.z,
            }
        }
    }

    impl Add for Quaternion {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self {
                w: self.w + rhs.w,
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
            }
        }
    }

    impl Mul<f64> for Quaternion {
        type Output = Self;

        fn mul(self, rhs: f64) -> Self::Output {
            Self {
                w: self.w * rhs,
                x: self.x * rhs,
                y: self.y * rhs,
                z: self.z * rhs,
            }
        }
    }
}

/// Handles and ticks that can safely cross the plugin boundary.
pub mod handles {
    /// Unique identifier for an entity within a dimension shard.
    #[derive(
        Debug,
        Clone,
        Copy,
        PartialEq,
        Eq,
        Hash,
        PartialOrd,
        Ord,
        serde::Serialize,
        serde::Deserialize,
    )]
    pub struct EntityId(pub u64);

    /// Identifier for a dimension.
    #[derive(
        Debug,
        Clone,
        Copy,
        PartialEq,
        Eq,
        Hash,
        PartialOrd,
        Ord,
        serde::Serialize,
        serde::Deserialize,
    )]
    pub struct DimensionId(pub u32);

    /// Logical tick counter used for deterministic scheduling.
    pub type Tick = u64;
}

/// Read-only world view exposed to plugins.
pub mod view {
    use super::handles::{DimensionId, Tick};

    /// Immutable snapshot of the active dimension state.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct WorldViewSnapshot {
        world_seed: u64,
        dimension: DimensionId,
        engine_tick: Tick,
        dimension_tick: Tick,
    }

    impl WorldViewSnapshot {
        /// Construct a view for a specific dimension with zeroed ticks.
        pub fn new(world_seed: u64, dimension: DimensionId) -> Self {
            Self {
                world_seed,
                dimension,
                engine_tick: 0,
                dimension_tick: 0,
            }
        }

        /// Construct a view that includes engine and dimension tick counters.
        pub fn with_ticks(
            world_seed: u64,
            dimension: DimensionId,
            engine_tick: Tick,
            dimension_tick: Tick,
        ) -> Self {
            Self {
                world_seed,
                dimension,
                engine_tick,
                dimension_tick,
            }
        }

        /// World seed driving all deterministic RNG streams.
        pub fn world_seed(&self) -> u64 {
            self.world_seed
        }

        /// Dimension being ticked.
        pub fn dimension(&self) -> DimensionId {
            self.dimension
        }

        /// Current engine tick.
        pub fn engine_tick(&self) -> Tick {
            self.engine_tick
        }

        /// Current tick within the active dimension.
        pub fn dimension_tick(&self) -> Tick {
            self.dimension_tick
        }
    }

    /// Read-only access to deterministic world metadata.
    pub trait WorldView {
        fn world_seed(&self) -> u64;
        fn dimension(&self) -> DimensionId;
        fn engine_tick(&self) -> Tick;
        fn dimension_tick(&self) -> Tick;
    }

    impl WorldView for WorldViewSnapshot {
        fn world_seed(&self) -> u64 {
            self.world_seed()
        }

        fn dimension(&self) -> DimensionId {
            self.dimension()
        }

        fn engine_tick(&self) -> Tick {
            self.engine_tick()
        }

        fn dimension_tick(&self) -> Tick {
            self.dimension_tick()
        }
    }
}

/// Intent envelopes that can be queued without exposing the engine world.
pub mod intent {
    use super::handles::{DimensionId, EntityId, Tick};

    /// Deterministic intent/command envelope.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct IntentEnvelope<P> {
        pub issued_by: EntityId,
        pub issued_at: Tick,
        pub applies_at: Tick,
        pub payload: P,
        pub dimension: Option<DimensionId>,
    }

    impl<P> IntentEnvelope<P> {
        /// Target dimension for the envelope, defaulting to the current view.
        pub fn target_dimension(&self, default: DimensionId) -> DimensionId {
            self.dimension.unwrap_or(default)
        }

        /// Override the target dimension.
        pub fn for_dimension(mut self, dimension: DimensionId) -> Self {
            self.dimension = Some(dimension);
            self
        }
    }
}

/// Plugin trait and engine adapter that keep third-party authors away from
/// `World` and `Scheduler`.
pub mod plugin {
    use std::any::Any;

    use super::handles::{DimensionId, Tick};
    use super::intent::IntentEnvelope;
    use super::view::WorldView;

    /// Narrow plugin trait for external authors.
    pub trait ContextPlugin: Any {
        /// Intent payload emitted by the plugin.
        type Intent: Clone + Send + 'static;

        /// Deterministic per-tick callback.
        fn on_tick(&mut self, world: &dyn WorldView) -> Vec<IntentEnvelope<Self::Intent>> {
            let _ = world;
            Vec::new()
        }

        /// Optional event hook.
        fn on_event(&mut self, _event: ContextEvent) -> Vec<IntentEnvelope<Self::Intent>> {
            Vec::new()
        }

        /// Downcast helper.
        fn as_any(&self) -> &dyn Any
        where
            Self: 'static + Sized,
        {
            self
        }

        /// Mutable downcast helper.
        fn as_any_mut(&mut self) -> &mut dyn Any
        where
            Self: 'static + Sized,
        {
            self
        }
    }

    /// Events routed to external plugins without exposing engine internals.
    #[derive(Debug, Clone)]
    pub enum ContextEvent {
        /// Engine tick occurred for a dimension.
        EngineTick {
            engine_tick: Tick,
            dimension: DimensionId,
        },
        /// Engine-specific custom event scoped to a dimension.
        Custom { id: u32, dimension: DimensionId },
    }
}

/// FFI-friendly ABI types shared with WASM plugins. These mirror the layouts
/// defined in the engine's `plugin_abi` module without requiring engine
/// dependencies.
pub mod abi {
    use bytemuck::{Pod, Zeroable};

    /// Status codes returned by host callbacks surfaced to WASM guests.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum HostCallStatus {
        Ok = 0,
        MissingWorld = 1,
        MissingMemory = 2,
        InvalidEntity = 3,
        IntentRejected = 4,
    }

    impl HostCallStatus {
        pub fn from_i32(value: i32) -> Self {
            match value {
                0 => HostCallStatus::Ok,
                1 => HostCallStatus::MissingWorld,
                2 => HostCallStatus::MissingMemory,
                3 => HostCallStatus::InvalidEntity,
                4 => HostCallStatus::IntentRejected,
                _ => HostCallStatus::IntentRejected,
            }
        }
    }

    /// Identifier for an entity within a specific dimension.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
    pub struct PluginEntity {
        pub id: u64,
        pub dimension: u32,
        /// Reserved for future flags; keeps the struct 16-byte aligned.
        pub reserved: u32,
    }

    impl PluginEntity {
        pub fn new(id: u64, dimension: u32) -> Self {
            Self {
                id,
                dimension,
                reserved: 0,
            }
        }
    }

    /// Simple vector type for FFI use.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
    pub struct PluginVec3 {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }

    impl PluginVec3 {
        pub fn zero() -> Self {
            Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }
        }
    }

    /// Minimal snapshot of an entity available to plugins.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
    pub struct PluginEntitySnapshot {
        pub entity: PluginEntity,
        pub has_transform: u8,
        pub has_velocity: u8,
        pub has_engine: u8,
        /// Reserved for future flags.
        pub reserved: u8,
        /// Padding to keep subsequent fields 8-byte aligned.
        pub padding: u32,
        pub position: PluginVec3,
        pub velocity: PluginVec3,
        pub throttle: f64,
    }

    impl PluginEntitySnapshot {
        pub fn empty(entity: PluginEntity) -> Self {
            Self {
                entity,
                has_transform: 0,
                has_velocity: 0,
                has_engine: 0,
                reserved: 0,
                padding: 0,
                position: PluginVec3::zero(),
                velocity: PluginVec3::zero(),
                throttle: 0.0,
            }
        }
    }

    /// Intent kinds surfaced to the host. These are interpreted deterministically
    /// by the validator registered for plugin intents.
    #[repr(i32)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PluginIntentKind {
        None = 0,
        SetThrottle = 1,
    }

    impl PluginIntentKind {
        pub fn from_i32(value: i32) -> Option<Self> {
            match value {
                0 => Some(PluginIntentKind::None),
                1 => Some(PluginIntentKind::SetThrottle),
                _ => None,
            }
        }
    }

    /// Intent envelope copied out of WASM guest memory.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
    pub struct PluginIntentEnvelope {
        pub target: PluginEntity,
        pub issued_by: PluginEntity,
        pub applies_at: u64,
        pub kind: i32,
        /// Reserved for future flags; keeps the struct 8-byte aligned.
        pub padding: u32,
        /// Generic scalar payload; currently used for throttle values.
        pub scalar: f64,
    }

    /// Tick context forwarded to plugins.
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable, Default)]
    pub struct PluginTickInfo {
        pub world_seed: u64,
        pub engine_tick: u64,
        pub dimension_tick: u64,
        pub dimension: u32,
        pub reserved: u32,
    }
}

/// Guest-side helpers for interacting with the host callbacks exposed by the
/// engine when running inside a WASM plugin.
#[cfg(target_arch = "wasm32")]
pub mod guest {
    use super::abi::{
        HostCallStatus, PluginEntity, PluginEntitySnapshot, PluginIntentEnvelope, PluginIntentKind,
    };

    #[link(wasm_import_module = "phase_space_host")]
    extern "C" {
        fn log(ptr: *const u8, len: u32);
        fn get_entity_snapshot(
            entity_id: u64,
            dimension: u32,
            out_ptr: *mut PluginEntitySnapshot,
        ) -> i32;
        fn find_first_entity_with_engine(out_ptr: *mut PluginEntitySnapshot) -> i32;
        fn submit_intent(intent_ptr: *const PluginIntentEnvelope) -> i32;
    }

    /// Write a UTF-8 log message through the host.
    pub fn host_log(message: &str) {
        unsafe { log(message.as_ptr(), message.len() as u32) };
    }

    /// Convenience helper to describe a throttle change intent.
    pub fn throttle_intent(
        entity: PluginEntity,
        applies_at: u64,
        throttle: f64,
    ) -> PluginIntentEnvelope {
        PluginIntentEnvelope {
            target: entity,
            issued_by: entity,
            applies_at,
            kind: PluginIntentKind::SetThrottle as i32,
            padding: 0,
            scalar: throttle,
        }
    }

    /// Fetch a snapshot for a specific entity from the host.
    pub fn host_get_entity_snapshot(
        entity: PluginEntity,
        out: &mut PluginEntitySnapshot,
    ) -> HostCallStatus {
        let status =
            unsafe { get_entity_snapshot(entity.id, entity.dimension, out as *mut _) } as i32;
        host_status_from_i32(status)
    }

    /// Ask the host for the first entity with an engine component.
    pub fn host_find_first_engine(out: &mut PluginEntitySnapshot) -> HostCallStatus {
        host_status_from_i32(unsafe { find_first_entity_with_engine(out as *mut _) })
    }

    /// Submit an intent envelope back to the host for validation.
    pub fn host_submit_intent(intent: &PluginIntentEnvelope) -> HostCallStatus {
        host_status_from_i32(unsafe { submit_intent(intent as *const _) })
    }

    fn host_status_from_i32(raw: i32) -> HostCallStatus {
        HostCallStatus::from_i32(raw)
    }
}

/// Stubbed guest helpers for non-WASM targets to keep the API ergonomic in
/// host-side tests.
#[cfg(not(target_arch = "wasm32"))]
pub mod guest {
    use super::abi::{HostCallStatus, PluginEntity, PluginEntitySnapshot, PluginIntentEnvelope};

    pub fn host_log(_message: &str) {}

    pub fn throttle_intent(
        entity: PluginEntity,
        applies_at: u64,
        throttle: f64,
    ) -> PluginIntentEnvelope {
        PluginIntentEnvelope {
            target: entity,
            issued_by: entity,
            applies_at,
            kind: super::abi::PluginIntentKind::SetThrottle as i32,
            padding: 0,
            scalar: throttle,
        }
    }

    pub fn host_get_entity_snapshot(
        _entity: PluginEntity,
        _out: &mut PluginEntitySnapshot,
    ) -> HostCallStatus {
        HostCallStatus::MissingWorld
    }

    pub fn host_find_first_engine(_out: &mut PluginEntitySnapshot) -> HostCallStatus {
        HostCallStatus::MissingWorld
    }

    pub fn host_submit_intent(_intent: &PluginIntentEnvelope) -> HostCallStatus {
        HostCallStatus::IntentRejected
    }
}

pub use handles::{DimensionId, EntityId, Tick};
pub use intent::IntentEnvelope;
pub use plugin::{ContextEvent, ContextPlugin};
pub use view::{WorldView, WorldViewSnapshot};
