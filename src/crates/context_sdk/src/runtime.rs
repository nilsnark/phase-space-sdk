//! Minimal scheduler/runtime-facing abstractions exposed through the SDK so contexts
//! can describe systems without depending on engine scheduler types.
use crate::context::{BrainDto, BrainState as BrainStateDto, DeterministicContextBlob};
use crate::handles::{DimensionId, EntityId};
use crate::sensors::{ScanRequest, SensorDef, SensorState};
use crate::determinism::ScriptRngState;
use crate::world::{MassDto, TransformDto, VelocityDto};
use crate::determinism::WorldSeed;
use crate::script::SelfView as ScriptSelfView;
use std::collections::HashMap;
use std::any::Any;

/// Declarative phase ordering for system execution within a dimension tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TickPhase {
    Sensors,
    Scripts,
    IntentCommit,
    Physics,
    /// Custom phase identifier for advanced pipelines.
    Custom(u8),
}

/// Stable identifier for a system within a phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SystemId(pub u64);

/// Explicit ordering of phases to execute for each dimension tick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseOrder(pub Vec<TickPhase>);

impl PhaseOrder {
    pub fn new(phases: Vec<TickPhase>) -> Self {
        let mut seen = HashMap::new();
        let mut ordered = Vec::with_capacity(phases.len());

        for phase in phases {
            if seen.insert(phase, ()).is_none() {
                ordered.push(phase);
            }
        }

        PhaseOrder(ordered)
    }

    pub fn iter(&self) -> impl Iterator<Item = TickPhase> + '_ {
        self.0.iter().copied()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Default for PhaseOrder {
    fn default() -> Self {
        PhaseOrder::new(vec![
            TickPhase::Physics,
            TickPhase::Sensors,
            TickPhase::Scripts,
            TickPhase::IntentCommit,
        ])
    }
}

/// Declarative scheduling profile for a dimension type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionProfile {
    pub phase_order: PhaseOrder,
    systems_by_phase: HashMap<TickPhase, Vec<SystemId>>,
}

impl DimensionProfile {
    pub fn new(phase_order: PhaseOrder) -> Self {
        Self {
            phase_order,
            systems_by_phase: HashMap::new(),
        }
    }

    pub fn register_system(&mut self, phase: TickPhase, system_id: SystemId) {
        self.systems_by_phase
            .entry(phase)
            .or_default()
            .push(system_id);
    }

    pub fn systems_for_phase(&self, phase: TickPhase) -> &[SystemId] {
        self.systems_by_phase
            .get(&phase)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

impl Default for DimensionProfile {
    fn default() -> Self {
        Self::new(PhaseOrder::default())
    }
}

/// Execution context supplied to systems each tick.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemRunContext {
    /// Dimension the system is executing in.
    pub dimension: DimensionId,
    /// Engine tick index (global).
    pub engine_tick: u64,
    /// Current local tick count within the dimension.
    pub dimension_tick: u64,
}

impl SystemRunContext {
    pub fn new(dimension: DimensionId, engine_tick: u64, dimension_tick: u64) -> Self {
        Self {
            dimension,
            engine_tick,
            dimension_tick,
        }
    }
}
impl Default for SystemRunContext {
    fn default() -> Self {
        Self {
            dimension: DimensionId(0),
            engine_tick: 0,
            dimension_tick: 0,
        }
    }
}

/// Narrow host surface available to systems at runtime. Engines are expected to
/// provide an implementation that applies mutations to the live world.
pub trait RuntimeWorld {
    /// Optional downcast helper for engine adapters.
    fn as_any(&self) -> Option<&dyn Any> {
        None
    }

    /// Deterministic world seed.
    fn world_seed(&self) -> WorldSeed {
        0
    }

    /// Enumerate entities in a dimension.
    fn entities_in_dimension(&self, dimension: DimensionId) -> Option<Vec<EntityId>> {
        let _ = dimension;
        None
    }

    /// Apply all intents that are ready for the current tick.
    fn apply_ready_intents(&mut self, _dimension: DimensionId, _tick: u64) {}

    /// Dimension tick duration in seconds.
    fn tick_seconds(&self, _dimension: DimensionId) -> f64 {
        1.0
    }

    /// Brain component for an entity.
    fn brain(&self, _dimension: DimensionId, _entity: EntityId) -> Option<BrainDto> {
        None
    }

    /// Update or attach a brain component (metadata).
    fn set_brain(&mut self, _dimension: DimensionId, _entity: EntityId, _brain: BrainDto) {}

    /// Brain state payload for an entity.
    fn brain_state(&self, _dimension: DimensionId, _entity: EntityId) -> Option<BrainStateDto> {
        None
    }

    /// Replace or attach a brain state.
    fn set_brain_state(&mut self, _dimension: DimensionId, _entity: EntityId, _state: BrainStateDto) {}

    /// Sensor definition for an entity.
    fn sensor_def(&self, _dimension: DimensionId, _entity: EntityId) -> Option<SensorDef> {
        None
    }

    /// Update or attach a sensor definition.
    fn set_sensor_def(&mut self, _dimension: DimensionId, _entity: EntityId, _def: SensorDef) {}

    /// Sensor state for an entity.
    fn sensor_state(&self, _dimension: DimensionId, _entity: EntityId) -> Option<SensorState> {
        None
    }

    /// Update or attach a sensor state.
    fn set_sensor_state(&mut self, _dimension: DimensionId, _entity: EntityId, _state: SensorState) {}

    /// Pending scan request for an entity.
    fn scan_request(&self, _dimension: DimensionId, _entity: EntityId) -> Option<ScanRequest> {
        None
    }

    /// Set or replace a scan request.
    fn set_scan_request(&mut self, _dimension: DimensionId, _entity: EntityId, _request: ScanRequest) {}

    /// Velocity for an entity.
    fn velocity(&self, _dimension: DimensionId, _entity: EntityId) -> Option<VelocityDto> {
        None
    }

    /// Update or attach velocity.
    fn set_velocity(&mut self, _dimension: DimensionId, _entity: EntityId, _velocity: VelocityDto) {}

    /// Transform for an entity.
    fn transform(&self, _dimension: DimensionId, _entity: EntityId) -> Option<TransformDto> {
        None
    }

    /// Update or attach a transform.
    fn set_transform(&mut self, _dimension: DimensionId, _entity: EntityId, _transform: TransformDto) {}

    /// Mass for an entity.
    fn mass(&self, _dimension: DimensionId, _entity: EntityId) -> Option<MassDto> {
        None
    }

    /// Update or attach a mass component.
    fn set_mass(&mut self, _dimension: DimensionId, _entity: EntityId, _mass: MassDto) {}

    /// Mailbox contents (inbox, outbox) for a brain.
    fn brain_mailbox(
        &self,
        _dimension: DimensionId,
        _entity: EntityId,
    ) -> Option<(Vec<DeterministicContextBlob>, Vec<DeterministicContextBlob>)> {
        None
    }

    /// Replace mailbox contents.
    fn set_brain_mailbox(
        &mut self,
        _dimension: DimensionId,
        _entity: EntityId,
        _inbox: Vec<DeterministicContextBlob>,
        _outbox: Vec<DeterministicContextBlob>,
    ) {
    }

    /// Read-only self view for the given entity.
    fn self_view(
        &self,
        _dimension: DimensionId,
        _entity: EntityId,
    ) -> Option<ScriptSelfView> {
        None
    }

    /// Access to a brain's script RNG state.
    fn script_rng(&self, _brain: EntityId) -> Option<ScriptRngState> {
        None
    }

    /// Mutably borrow a brain's script RNG state.
    fn script_rng_mut(&mut self, _brain: EntityId) -> Option<&mut ScriptRngState> {
        None
    }
}

/// System trait that contexts can implement without depending on engine scheduler
/// types. Engines can adapt this to their internal scheduler.
pub trait System {
    fn run(&mut self, world: &mut dyn RuntimeWorld, dt: f64, ctx: &mut SystemRunContext);
}
