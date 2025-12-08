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

/// Minimal world/scheduler facade for SDK consumers.
pub mod world {
    use super::context::{Actor, BrainDto, BrainState};
    use super::handles::{DimensionId, EntityId};
    use super::math::{Quaternion, Vec3};
    use super::sensors::{ScanRequest, SensorDef, SensorState};

    /// Basic transform DTO.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TransformDto {
        pub position_m: Vec3,
    }

    impl TransformDto {
        pub fn new(position_m: Vec3) -> Self {
            Self { position_m }
        }
    }

    /// Basic velocity DTO.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct VelocityDto {
        pub linear_velocity_m_per_s: Vec3,
    }

    impl VelocityDto {
        pub fn new(linear_velocity_m_per_s: Vec3) -> Self {
            Self {
                linear_velocity_m_per_s,
            }
        }
    }

    /// Minimal dimension tag helper covering the standard engine dimension types.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum DimensionTag {
        Interstellar,
        Interplanetary,
        Orbital,
        Surface,
        Interior,
        Pocket,
    }

    impl DimensionTag {
        pub fn dimension_type_id(&self) -> &'static str {
            match self {
                DimensionTag::Interstellar => "core.dim.interstellar",
                DimensionTag::Interplanetary => "core.dim.interplanetary",
                DimensionTag::Orbital => "core.dim.orbital",
                DimensionTag::Surface => "core.dim.surface",
                DimensionTag::Interior => "core.dim.interior",
                DimensionTag::Pocket => "core.dim.pocket",
            }
        }
    }

    /// Simple frame reference DTO.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FrameRef {
        pub frame_id: u64,
    }

    impl FrameRef {
        pub fn new(frame_id: u64) -> Self {
            Self { frame_id }
        }

        pub fn root() -> Self {
            Self { frame_id: 0 }
        }
    }

    impl Default for FrameRef {
        fn default() -> Self {
            Self::root()
        }
    }

    /// Inertia tensor DTO represented along principal axes.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct InertiaTensorDto {
        pub ixx: f64,
        pub iyy: f64,
        pub izz: f64,
    }

    impl InertiaTensorDto {
        pub fn new(ixx: f64, iyy: f64, izz: f64) -> Self {
            Self { ixx, iyy, izz }
        }

        pub fn sphere(mass: f64, radius: f64) -> Self {
            let i = 0.4 * mass * radius * radius;
            Self {
                ixx: i,
                iyy: i,
                izz: i,
            }
        }

        pub fn point_mass() -> Self {
            Self {
                ixx: 1e-10,
                iyy: 1e-10,
                izz: 1e-10,
            }
        }
    }

    /// Basic mass properties DTO.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct MassPropertiesDto {
        pub mass: f64,
        pub inertia: InertiaTensorDto,
    }

    impl MassPropertiesDto {
        pub fn new(mass: f64, inertia: InertiaTensorDto) -> Self {
            Self { mass, inertia }
        }

        pub fn sphere(mass: f64, radius: f64) -> Self {
            Self {
                mass,
                inertia: InertiaTensorDto::sphere(mass, radius),
            }
        }

        pub fn point_mass(mass: f64) -> Self {
            Self {
                mass,
                inertia: InertiaTensorDto::point_mass(),
            }
        }
    }

    /// Phase space kinematics DTO mirroring the engine component.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PhaseKinematicsDto {
        pub position: Vec3,
        pub momentum: Vec3,
        pub orientation: Quaternion,
        pub angular_momentum: Vec3,
    }

    impl PhaseKinematicsDto {
        pub fn new(
            position: Vec3,
            momentum: Vec3,
            orientation: Quaternion,
            angular_momentum: Vec3,
        ) -> Self {
            Self {
                position,
                momentum,
                orientation: orientation.normalized(),
                angular_momentum,
            }
        }

        pub fn stationary() -> Self {
            Self {
                position: Vec3::zero(),
                momentum: Vec3::zero(),
                orientation: Quaternion::identity(),
                angular_momentum: Vec3::zero(),
            }
        }

        pub fn from_position_velocity(position: Vec3, velocity: Vec3, mass: f64) -> Self {
            Self {
                position,
                momentum: velocity * mass,
                orientation: Quaternion::identity(),
                angular_momentum: Vec3::zero(),
            }
        }

        pub fn velocity(&self, mass: f64) -> Vec3 {
            if mass > f64::EPSILON {
                self.momentum / mass
            } else {
                Vec3::zero()
            }
        }
    }

    /// Narrow host interface used by SDK-facing code to build worlds.
    pub trait WorldBuilder {
        fn world_seed(&self) -> u64;
        fn spawn_in(&mut self, dimension: DimensionId) -> EntityId;
        fn set_phase_kinematics(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            kinematics: PhaseKinematicsDto,
        );
        fn set_transform(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            transform: TransformDto,
        );
        fn set_velocity(&mut self, dimension: DimensionId, entity: EntityId, velocity: VelocityDto);
        fn set_mass_properties_sphere(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            mass_kg: f64,
            radius_m: f64,
        );
        fn set_mass(&mut self, dimension: DimensionId, entity: EntityId, mass_kg: f64);
        fn add_gravity_source(&mut self, dimension: DimensionId, entity: EntityId, mass_kg: f64);
        fn add_on_rails(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            radius_m: f64,
            angular_velocity_rad_s: f64,
            phase_rad: f64,
        );
        fn add_orbit_body(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            mass_kg: f64,
            primary_hint_entity: Option<u64>,
        );
        fn add_fuel_tank(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            fuel_type_id: u32,
            max_capacity_kg: f64,
            current_mass_kg: f64,
            consumption_enabled: bool,
        );
        fn add_engine(&mut self, dimension: DimensionId, entity: EntityId, engine: EngineDto);
        fn add_dimension_tag(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            dimension_type_id: &str,
        );
        fn add_frame_ref(&mut self, dimension: DimensionId, entity: EntityId, frame: FrameRef);
        fn add_actor(&mut self, dimension: DimensionId, entity: EntityId, actor: Actor);
        fn add_brain(&mut self, dimension: DimensionId, entity: EntityId, brain: BrainDto);
        fn add_brain_state(&mut self, dimension: DimensionId, entity: EntityId, state: BrainState);
        fn add_sensor_def(&mut self, dimension: DimensionId, entity: EntityId, def: SensorDef);
        fn add_sensor_state(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            state: SensorState,
        );
        fn add_scan_request(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            request: ScanRequest,
        );
    }

    /// Simple engine DTO mirroring the core engine component.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct EngineDto {
        pub max_thrust_n: f64,
        pub max_vector_angle_rad: f64,
        pub specific_impulse_s: f64,
        pub throttle: f64,
        pub active: bool,
        pub vector_angle_rad: f64,
    }

    /// Fuel tank component DTO exposed to contexts.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct FuelTankDto {
        pub fuel_type_id: u32,
        pub max_capacity_kg: f64,
        pub current_mass_kg: f64,
        pub consumption_enabled: bool,
    }

    /// Mass component DTO exposed to contexts.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct MassDto {
        pub mass: f64,
    }

    /// Maneuver node DTO mirroring the engine's maneuver plan node.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct ManeuverNodeDto {
        pub start_time_s: f64,
        pub duration_s: f64,
        pub thrust_x: f64,
        pub thrust_y: f64,
        pub is_delta_v: bool,
        pub accumulated_burn_time_s: f64,
    }
}

/// Deterministic RNG helpers mirrored from the engine for procgen and scripts.
pub mod determinism {
    use rand::{RngCore, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    /// Canonical seed type used throughout deterministic generation.
    pub type WorldSeed = u64;

    /// Distinct deterministic RNG streams grouped by purpose.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum RngDomain {
        /// Core engine systems such as physics integrators.
        Engine,
        /// Sensor updates and perception noise live here.
        Sensors,
        /// Scripting and gameplay logic hooks.
        Scripts,
        /// Procedural content generation.
        Procgen,
        /// Physics noise and perturbations.
        PhysicsNoise,
        /// User interface interactions.
        Ui,
    }

    impl RngDomain {
        #[allow(dead_code)]
        const _ORDERED: [RngDomain; 6] = [
            RngDomain::Engine,
            RngDomain::Sensors,
            RngDomain::Scripts,
            RngDomain::Procgen,
            RngDomain::PhysicsNoise,
            RngDomain::Ui,
        ];

        fn salt(self) -> u64 {
            match self {
                RngDomain::Engine => 0xC001_CAFE_D00D_0000,
                RngDomain::Sensors => 0xC001_CAFE_D00D_0001,
                RngDomain::Scripts => 0xC001_CAFE_D00D_0002,
                RngDomain::Procgen => 0xC001_CAFE_D00D_0003,
                RngDomain::PhysicsNoise => 0xC001_CAFE_D00D_0004,
                RngDomain::Ui => 0xC001_CAFE_D00D_0005,
            }
        }
    }

    fn splitmix64(mut state: u64) -> u64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fold_mix(state: u64, value: u64, tweak: u64) -> u64 {
        splitmix64(state ^ value.wrapping_mul(tweak))
    }

    /// Stateless, keyed RNG suitable for procgen domains that must not depend on call order.
    pub fn engine_rand_u64(seed: WorldSeed, domain: RngDomain, a: u64, b: u64, c: u64) -> u64 {
        let mut state = splitmix64(seed ^ domain.salt());
        state = fold_mix(state, a, 0x9E37_79B9);
        state = fold_mix(state, b, 0xC2B2_AE35);
        state = fold_mix(state, c, 0x1656_67B1);
        splitmix64(state ^ 0xD1B5_4A32_4F3A_9E55)
    }

    /// Generate a deterministic f64 in the half-open range [0, 1).
    pub fn engine_rand_f64(seed: WorldSeed, domain: RngDomain, a: u64, b: u64, c: u64) -> f64 {
        let raw = engine_rand_u64(seed, domain, a, b, c);
        let mantissa = raw >> 11;
        mantissa as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Stateful RNG used by scripts, keyed per brain.
    #[derive(Debug, Clone, PartialEq)]
    #[repr(transparent)]
    pub struct ScriptRngState {
        rng: Pcg64Mcg,
    }

    impl ScriptRngState {
        /// Construct a script RNG from a 128-bit seed.
        pub fn from_seed(seed: u128) -> Self {
            Self {
                rng: Pcg64Mcg::from_seed(seed.to_le_bytes()),
            }
        }

        /// Reconstruct a script RNG from an existing generator state.
        pub fn from_raw(rng: Pcg64Mcg) -> Self {
            Self { rng }
        }

        /// Borrow the underlying generator.
        pub fn as_raw(&self) -> &Pcg64Mcg {
            &self.rng
        }

        /// Mutably borrow the underlying generator.
        pub fn as_raw_mut(&mut self) -> &mut Pcg64Mcg {
            &mut self.rng
        }

        /// Take ownership of the underlying generator.
        pub fn into_raw(self) -> Pcg64Mcg {
            self.rng
        }

        /// Clone the underlying generator state.
        pub fn clone_raw(&self) -> Pcg64Mcg {
            self.rng.clone()
        }

        /// Produce the next deterministic u64.
        pub fn next_u64(&mut self) -> u64 {
            self.rng.next_u64()
        }

        /// Produce the next deterministic f64 in the half-open range [0, 1).
        pub fn next_f64(&mut self) -> f64 {
            let mantissa = self.next_u64() >> 11;
            mantissa as f64 * (1.0 / (1u64 << 53) as f64)
        }
    }
}

/// Sensor DTOs shared with contexts without exposing engine internals.
pub mod sensors {
    use super::handles::{DimensionId, EntityId, Tick};
    use super::math::Vec3;
    use super::view::WorldViewSnapshot;
    use crate::context::BrainId;
    use crate::determinism::{engine_rand_u64, RngDomain};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    /// Identifier for a sensor definition attached to an entity.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct SensorId(pub u64);

    /// Identifier for a frequency band or modality.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct BandId(pub u32);

    /// Identifier for an observation category understood by the active context.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ObservationKind(pub u32);

    /// Opaque resolution descriptor used to hint at expected measurement fidelity.
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Resolution {
        pub range_resolution: f32,
        pub angular_resolution: f32,
    }

    impl Resolution {
        pub fn new(range_resolution: f32, angular_resolution: f32) -> Self {
            Self {
                range_resolution,
                angular_resolution,
            }
        }
    }

    /// Static sensor definition.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SensorDef {
        pub sensor_id: SensorId,
        pub band: BandId,
        pub aperture: f32,
        pub fov: f32,
        pub max_range: f32,
        pub resolution: Resolution,
        pub integration_ticks: u32,
        pub power_cost_kw: f32,
        pub cooldown_ticks: u32,
    }

    /// Runtime state maintained for each sensor.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SensorState {
        pub cooldown_remaining: u32,
        pub integrating_since: Option<Tick>,
    }

    impl SensorState {
        pub fn idle() -> Self {
            Self {
                cooldown_remaining: 0,
                integrating_since: None,
            }
        }
    }

    impl Default for SensorState {
        fn default() -> Self {
            Self::idle()
        }
    }

    /// Command describing a directional scan.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ScanRequest {
        pub sensor_id: SensorId,
        pub direction: [f32; 3],
        pub integration_ticks: Option<u32>,
        pub target_hint: Option<EntityId>,
    }

    impl ScanRequest {
        pub fn new(sensor_id: SensorId, direction: Vec3) -> Self {
            Self {
                sensor_id,
                direction: [direction.x as f32, direction.y as f32, direction.z as f32],
                integration_ticks: None,
                target_hint: None,
            }
        }

        pub fn with_integration_ticks(mut self, ticks: u32) -> Self {
            self.integration_ticks = Some(ticks);
            self
        }

        pub fn with_target_hint(mut self, entity: EntityId) -> Self {
            self.target_hint = Some(entity);
            self
        }
    }

    /// Additional metadata carried alongside an observation.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ObservationMeta {
        pub band: BandId,
        pub integration_ticks: u32,
        pub aperture: f32,
        pub fov: f32,
        pub max_range: f32,
        pub resolution: Resolution,
        pub scan_direction: [f32; 3],
    }

    /// Final observation emitted by a sensor.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Observation {
        pub sensor_id: SensorId,
        pub time_tick: Tick,
        pub kind: ObservationKind,
        pub payload: Vec<f32>,
        pub snr: f32,
        pub target_hint: Option<EntityId>,
        pub meta: ObservationMeta,
    }

    /// Ideal, noise-free measurement computed by a context-specific closure.
    #[derive(Debug, Clone, PartialEq)]
    pub struct IdealObservation {
        pub kind: ObservationKind,
        pub payload: Vec<f32>,
        pub snr: f32,
        pub target_hint: Option<EntityId>,
    }

    impl IdealObservation {
        pub fn new(kind: ObservationKind, payload: Vec<f32>, snr: f32) -> Self {
            Self {
                kind,
                payload,
                snr,
                target_hint: None,
            }
        }

        pub fn with_target_hint(mut self, entity: EntityId) -> Self {
            self.target_hint = Some(entity);
            self
        }
    }

    /// View of a single target considered by a sensor.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Target {
        pub entity: EntityId,
        pub range: f32,
        pub line_of_sight: Vec3,
    }

    /// Minimal host API required by the SDK sensor system.
    pub trait SensorWorld {
        fn world_seed(&self) -> u64;
        fn entities_in_dimension(&self, dimension: DimensionId) -> Vec<EntityId>;
        fn transform_position(&self, dimension: DimensionId, entity: EntityId) -> Option<Vec3>;
        fn actor_brain(&self, dimension: DimensionId, entity: EntityId) -> Option<BrainId>;
        fn sensor_def(&self, dimension: DimensionId, entity: EntityId) -> Option<SensorDef>;
        fn scan_request(&self, dimension: DimensionId, entity: EntityId) -> Option<ScanRequest>;
        fn take_sensor_state(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
        ) -> Option<SensorState>;
        fn set_sensor_state(
            &mut self,
            dimension: DimensionId,
            entity: EntityId,
            state: SensorState,
        );
    }

    /// Deterministic sensor processing system that operates on [`SensorWorld`].
    pub struct SensorSystem<F>
    where
        F: Fn(&WorldViewSnapshot, &SensorDef, &Target) -> IdealObservation,
    {
        compute_observation: F,
        observation_buffers: HashMap<BrainId, Vec<Observation>>,
        payload_noise_scale: f32,
    }

    impl<F> SensorSystem<F>
    where
        F: Fn(&WorldViewSnapshot, &SensorDef, &Target) -> IdealObservation,
    {
        pub fn new(compute_observation: F) -> Self {
            Self {
                compute_observation,
                observation_buffers: HashMap::new(),
                payload_noise_scale: 0.01,
            }
        }

        /// Retrieve buffered observations for a specific brain, draining the buffer.
        pub fn drain_observations(&mut self, brain: BrainId) -> Vec<Observation> {
            self.observation_buffers.remove(&brain).unwrap_or_default()
        }

        /// Access all buffered observations without draining.
        pub fn observations(&self) -> &HashMap<BrainId, Vec<Observation>> {
            &self.observation_buffers
        }

        fn sample_noise(
            seed: u64,
            dimension: DimensionId,
            tick: Tick,
            counter: u64,
            scale: f32,
        ) -> f32 {
            let raw = engine_rand_u64(seed, RngDomain::Sensors, dimension.0 as u64, tick, counter);
            let unit = (raw >> 11) as f32 * (1.0 / (1u64 << 53) as f32);
            (unit - 0.5) * 2.0 * scale
        }

        fn add_noise(
            &self,
            seed: u64,
            dimension: DimensionId,
            tick: Tick,
            counter: &mut u64,
            ideal: &IdealObservation,
        ) -> Observation {
            let mut payload = Vec::with_capacity(ideal.payload.len());
            for &value in &ideal.payload {
                *counter += 1;
                let noise =
                    Self::sample_noise(seed, dimension, tick, *counter, self.payload_noise_scale);
                payload.push(value + noise);
            }

            *counter += 1;
            let snr_noise =
                Self::sample_noise(seed, dimension, tick, *counter, self.payload_noise_scale);

            Observation {
                sensor_id: SensorId(0),
                time_tick: tick,
                kind: ideal.kind,
                payload,
                snr: (ideal.snr + snr_noise).max(0.0),
                target_hint: ideal.target_hint,
                meta: ObservationMeta {
                    band: BandId(0),
                    integration_ticks: 0,
                    aperture: 0.0,
                    fov: 0.0,
                    max_range: 0.0,
                    resolution: Resolution::new(0.0, 0.0),
                    scan_direction: [0.0, 0.0, 0.0],
                },
            }
        }

        fn make_target(sensor_pos: Vec3, entity: EntityId, target_pos: Vec3) -> Target {
            let delta = target_pos - sensor_pos;
            let range = delta.length();
            let line_of_sight = if range > 0.0 {
                delta / range
            } else {
                Vec3::zero()
            };

            Target {
                entity,
                range: range as f32,
                line_of_sight,
            }
        }

        fn in_fov(direction: Vec3, los: Vec3, fov: f32) -> bool {
            let dir = direction.normalized();
            let cos_angle = dir.dot(los.normalized());
            let half_angle = fov as f64 * 0.5;
            cos_angle >= half_angle.cos()
        }

        pub fn run(
            &mut self,
            world: &mut dyn SensorWorld,
            dimension: DimensionId,
            engine_tick: Tick,
            dimension_tick: Tick,
        ) {
            let view = WorldViewSnapshot::with_ticks(
                world.world_seed(),
                dimension,
                engine_tick,
                dimension_tick,
            );
            let entities = world.entities_in_dimension(dimension);
            let seed = world.world_seed();

            for entity in entities.iter().copied() {
                let (Some(def), Some(request)) = (
                    world.sensor_def(dimension, entity),
                    world.scan_request(dimension, entity),
                ) else {
                    continue;
                };

                let mut state = world
                    .take_sensor_state(dimension, entity)
                    .unwrap_or_else(SensorState::idle);

                if state.cooldown_remaining > 0 {
                    state.cooldown_remaining -= 1;
                    world.set_sensor_state(dimension, entity, state);
                    continue;
                }

                let sensor_pos = match world.transform_position(dimension, entity) {
                    Some(pos) => pos,
                    None => continue,
                };

                let start_tick = state.integrating_since.get_or_insert(dimension_tick);
                let required_ticks =
                    request.integration_ticks.unwrap_or(def.integration_ticks) as Tick;
                if dimension_tick - *start_tick + 1 < required_ticks {
                    world.set_sensor_state(dimension, entity, state);
                    continue;
                }

                state.integrating_since = None;
                state.cooldown_remaining = def.cooldown_ticks;

                let scan_dir = Vec3::new(
                    request.direction[0] as f64,
                    request.direction[1] as f64,
                    request.direction[2] as f64,
                );

                let mut counter = 0;
                for target_entity in entities.iter().copied().filter(|e| *e != entity) {
                    let Some(target_pos) = world.transform_position(dimension, target_entity)
                    else {
                        continue;
                    };

                    let target = Self::make_target(sensor_pos, target_entity, target_pos);
                    if target.range as f64 > def.max_range as f64
                        || !Self::in_fov(scan_dir, target.line_of_sight, def.fov)
                    {
                        continue;
                    }

                    let mut observation = (self.compute_observation)(&view, &def, &target);
                    if observation.target_hint.is_none() {
                        observation.target_hint = Some(target.entity);
                    }

                    let mut noisy =
                        self.add_noise(seed, dimension, dimension_tick, &mut counter, &observation);
                    noisy.sensor_id = def.sensor_id;
                    noisy.meta = ObservationMeta {
                        band: def.band,
                        integration_ticks: request
                            .integration_ticks
                            .unwrap_or(def.integration_ticks),
                        aperture: def.aperture,
                        fov: def.fov,
                        max_range: def.max_range,
                        resolution: def.resolution,
                        scan_direction: request.direction,
                    };

                    if let Some(brain) = world.actor_brain(dimension, entity) {
                        self.observation_buffers
                            .entry(brain)
                            .or_default()
                            .push(noisy);
                    }
                }

                world.set_sensor_state(dimension, entity, state);
            }
        }
    }
}

/// Context-facing deterministic identifiers and payload containers.
pub mod context {
    use serde::{Deserialize, Serialize};

    /// Identifier for a deterministic brain driving an actor.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
    pub struct BrainId(pub u64);

    /// Opaque deterministic blob passed between contexts and the engine.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
    pub struct DeterministicContextBlob(pub Vec<u8>);

    impl DeterministicContextBlob {
        pub fn new(bytes: Vec<u8>) -> Self {
            Self(bytes)
        }

        pub fn as_slice(&self) -> &[u8] {
            self.0.as_slice()
        }
    }

    /// Brain-local state attached to an actor.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BrainState {
        pub payload: DeterministicContextBlob,
    }

    impl BrainState {
        pub fn new(payload: DeterministicContextBlob) -> Self {
            Self { payload }
        }

        pub fn clear(&mut self) {
            self.payload = DeterministicContextBlob::default();
        }
    }

    /// Marker component tagging an entity as being driven by a brain.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Actor {
        pub brain: BrainId,
    }

    impl Actor {
        pub fn new(brain: BrainId) -> Self {
            Self { brain }
        }
    }

    /// Snapshot of a deterministic brain used for initialization.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BrainDto {
        pub id: BrainId,
        pub last_engine_tick: u64,
        pub last_dimension_tick: u64,
    }

    impl BrainDto {
        pub fn new(id: BrainId) -> Self {
            Self {
                id,
                last_engine_tick: 0,
                last_dimension_tick: 0,
            }
        }

        pub fn with_ticks(mut self, engine_tick: u64, dimension_tick: u64) -> Self {
            self.last_engine_tick = engine_tick;
            self.last_dimension_tick = dimension_tick;
            self
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

pub use context::{Actor, BrainId, BrainState, DeterministicContextBlob};
pub use determinism::{engine_rand_f64, engine_rand_u64, RngDomain, ScriptRngState, WorldSeed};
pub use handles::{DimensionId, EntityId, Tick};
pub use intent::IntentEnvelope;
pub use plugin::{ContextEvent, ContextPlugin};
pub use script::{
    BrainRef as ScriptBrainRef, Mailbox as ScriptMailbox, ScriptContextBuilder, ScriptError,
    ScriptId, ScriptResult, ScriptRuntime, SelfView as ScriptSelfView,
};
pub use runtime::*;
pub use sensors::{
    BandId, IdealObservation, Observation, ObservationKind, ObservationMeta, Resolution,
    ScanRequest, SensorDef, SensorId, SensorState, SensorSystem, SensorWorld, Target,
};
pub use view::{WorldView, WorldViewSnapshot};
pub use world::{
    DimensionTag, EngineDto, FrameRef, InertiaTensorDto, MassPropertiesDto, PhaseKinematicsDto,
    TransformDto, VelocityDto, WorldBuilder,
};

pub mod runtime;

/// Scripting utilities and DTOs surfaced to contexts without exposing engine internals.
pub mod script {
    use crate::context::{Actor, BrainId, BrainState, DeterministicContextBlob};
    use crate::handles::{DimensionId, EntityId};
    use crate::world::{EngineDto, FuelTankDto, MassDto, TransformDto, VelocityDto};

    /// Stable identifier for a script.
    pub type ScriptId = u64;

    /// Simple mailbox DTO carrying inbox/outbox blobs.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct Mailbox {
        pub inbox: Vec<DeterministicContextBlob>,
        pub outbox: Vec<DeterministicContextBlob>,
    }

    impl Mailbox {
        pub fn new(inbox: Vec<DeterministicContextBlob>, outbox: Vec<DeterministicContextBlob>) -> Self {
            Self { inbox, outbox }
        }
    }

    /// Reference to a brain within a specific dimension/entity.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BrainRef {
        pub brain_id: BrainId,
        pub entity: EntityId,
        pub dimension: DimensionId,
    }

    impl BrainRef {
        pub fn new(brain_id: BrainId, entity: EntityId, dimension: DimensionId) -> Self {
            Self {
                brain_id,
                entity,
                dimension,
            }
        }
    }

    /// Errors surfaced by scripting runtimes.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ScriptError {
        InvalidParameter(String),
        MissingScript(ScriptId),
        BudgetExceeded,
    }

    /// Convenience alias for scripting results.
    pub type ScriptResult<T> = Result<T, ScriptError>;

    /// Build a per-brain scripting context from a runtime world.
    pub trait ScriptContextBuilder {
        type Ctx;

        fn build_ctx(
            &self,
            world: &dyn crate::runtime::RuntimeWorld,
            brain: BrainRef,
            brain_state: Option<BrainState>,
            mailbox: Option<Mailbox>,
        ) -> Self::Ctx;
    }

    /// Minimal runtime interface for loading and ticking scripts.
    pub trait ScriptRuntime {
        type Ctx;

        fn load_script(&mut self, id: ScriptId, source: &str) -> ScriptResult<()>;
        fn init(&mut self, id: ScriptId, ctx: &Self::Ctx) -> ScriptResult<()>;
        fn tick(&mut self, id: ScriptId, ctx: &Self::Ctx) -> ScriptResult<()>;
    }

    /// Snapshot of the entity that owns a brain, expressed in SDK DTOs.
    #[derive(Debug, Clone, PartialEq)]
    pub struct SelfView {
        pub entity: EntityId,
        pub dimension: DimensionId,
        pub actor: Option<Actor>,
        pub transform: Option<TransformDto>,
        pub velocity: Option<VelocityDto>,
        pub mass: Option<MassDto>,
        pub engine: Option<EngineDto>,
        pub fuel_tank: Option<FuelTankDto>,
    }

    impl SelfView {
        pub fn new(entity: EntityId, dimension: DimensionId) -> Self {
            Self {
                entity,
                dimension,
                actor: None,
                transform: None,
                velocity: None,
                mass: None,
                engine: None,
                fuel_tank: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::determinism::*;

    #[test]
    fn engine_rand_is_stateless() {
        let seed: WorldSeed = 123_456;
        let domain = RngDomain::Procgen;

        let first = engine_rand_u64(seed, domain, 1, 2, 3);
        let _noise = engine_rand_u64(seed, RngDomain::Sensors, 9, 8, 7);
        let second = engine_rand_u64(seed, domain, 1, 2, 3);

        assert_eq!(first, second);
    }

    #[test]
    fn script_rng_matches_float_range() {
        let mut rng = ScriptRngState::from_seed(0xABCD);
        let value = rng.next_f64();
        assert!(value >= 0.0 && value < 1.0);
    }
}
