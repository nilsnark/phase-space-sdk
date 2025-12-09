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

/// Canonical tick type used by schedulers and systems.
pub type Tick = u64;

/// Declarative phase ordering for system execution within a dimension tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug, PartialEq)]
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

impl TickPhase {
    /// Map phases to stable indices used by the simple scheduler.
    pub fn as_index(&self) -> usize {
        match self {
            TickPhase::Sensors => 0,
            TickPhase::Scripts => 1,
            TickPhase::IntentCommit => 2,
            TickPhase::Physics => 3,
            TickPhase::Custom(_) => 3,
        }
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

    /// Mutable downcast helper for engine adapters.
    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
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

/// Deterministic duration (microseconds) used for SDK-only schedulers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct SimDuration(pub u64);

impl SimDuration {
    pub const ZERO: SimDuration = SimDuration(0);

    pub fn from_micros(micros: u64) -> Self {
        SimDuration(micros)
    }

    pub fn from_secs_f64(secs: f64) -> Self {
        SimDuration((secs * 1_000_000.0).round() as u64)
    }

    pub fn as_micros(self) -> u64 {
        self.0
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
}

/// Engine clock DTO for SDK schedulers.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EngineClock {
    pub tick: u64,
    pub tick_duration: SimDuration,
    pub current_instant_micros: u64,
}

impl EngineClock {
    pub fn new(tick_duration: SimDuration) -> Self {
        Self {
            tick: 0,
            tick_duration,
            current_instant_micros: 0,
        }
    }

    pub fn step(&mut self) {
        self.tick += 1;
        self.current_instant_micros = self.current_instant_micros.saturating_add(self.tick_duration.as_micros());
    }

    pub fn elapsed(&self) -> SimDuration {
        SimDuration(self.current_instant_micros)
    }
}

/// Clock for a single dimension.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DimensionClock {
    pub id: DimensionId,
    pub time_scale: f64,
    pub local_tick_duration: SimDuration,
    pub local_time: SimDuration,
    pub accumulator: SimDuration,
    pub pending_ticks: u64,
    pub local_tick: u64,
}

impl DimensionClock {
    pub fn new(id: DimensionId, time_scale: f64, local_tick_duration: SimDuration) -> Self {
        Self {
            id,
            time_scale,
            local_tick_duration,
            local_time: SimDuration::ZERO,
            accumulator: SimDuration::ZERO,
            pending_ticks: 0,
            local_tick: 0,
        }
    }

    pub fn accumulate_engine_step(&mut self, engine_dt: SimDuration) -> u64 {
        if self.time_scale <= 0.0 {
            return 0;
        }
        let scaled = (engine_dt.as_micros() as f64 * self.time_scale).round() as u64;
        let scaled = SimDuration::from_micros(scaled);
        self.accumulator = SimDuration(self.accumulator.as_micros().saturating_add(scaled.as_micros()));
        self.local_time = SimDuration(self.local_time.as_micros().saturating_add(scaled.as_micros()));

        let tick_us = self.local_tick_duration.as_micros();
        if tick_us == 0 {
            return 0;
        }
        let due_total = self.accumulator.as_micros() / tick_us;
        let new_ticks = due_total.saturating_sub(self.pending_ticks);
        if new_ticks > 0 {
            self.pending_ticks = self.pending_ticks.saturating_add(new_ticks);
        }
        new_ticks
    }

    pub fn consume_ticks(&mut self, max_ticks: u64) -> u64 {
        if max_ticks == 0 || self.pending_ticks == 0 {
            return 0;
        }
        let to_run = self.pending_ticks.min(max_ticks);
        self.pending_ticks -= to_run;
        self.local_tick = self.local_tick.saturating_add(to_run);
        let consumed = self.local_tick_duration.as_micros().saturating_mul(to_run);
        let remaining = self.accumulator.as_micros().saturating_sub(consumed);
        self.accumulator = SimDuration(remaining);
        to_run
    }
}

/// Scheduling profile for a dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct DimensionSchedule {
    pub clock: DimensionClock,
    pub priority: i32,
    pub tick_budget_per_step: u64,
    pub fixed_timestep: Option<SimDuration>,
    pub paused: bool,
    pub effective_tick_rate_override: Option<f64>,
    pub profile: DimensionProfile,
}

impl DimensionSchedule {
    pub fn new(
        clock: DimensionClock,
        priority: i32,
        tick_budget_per_step: u64,
        fixed_timestep: Option<SimDuration>,
        paused: bool,
        effective_tick_rate_override: Option<f64>,
        profile: DimensionProfile,
    ) -> Self {
        Self {
            clock,
            priority,
            tick_budget_per_step,
            fixed_timestep,
            paused,
            effective_tick_rate_override,
            profile,
        }
    }

    fn ordering_key(&self) -> (i32, u32) {
        (self.priority, self.clock.id.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedTick {
    pub dimension: DimensionId,
    pub tick_index: u64,
    pub priority: i32,
    pub dt: SimDuration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedPhase {
    pub dimension: DimensionId,
    pub tick_index: u64,
    pub priority: i32,
    pub dt: SimDuration,
    pub phase: TickPhase,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TickSchedule {
    pub engine_tick: u64,
    pub ticks: Vec<ResolvedTick>,
    pub phases: Vec<ResolvedPhase>,
}

/// Simple engine scheduler that advances dimension clocks deterministically.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulationScheduler {
    pub engine_clock: EngineClock,
    pub dimensions: Vec<DimensionSchedule>,
}

impl SimulationScheduler {
    pub fn new(engine_clock: EngineClock) -> Self {
        Self {
            engine_clock,
            dimensions: Vec::new(),
        }
    }

    pub fn add_dimension(&mut self, schedule: DimensionSchedule) {
        let idx = self
            .dimensions
            .binary_search_by(|existing| existing.ordering_key().cmp(&schedule.ordering_key()))
            .unwrap_or_else(|idx| idx);
        self.dimensions.insert(idx, schedule);
    }

    pub fn step(&mut self) -> Vec<(DimensionId, u64)> {
        self.engine_clock.step();
        let engine_dt = self.engine_clock.tick_duration;
        self.dimensions
            .iter_mut()
            .map(|schedule| {
                if schedule.paused {
                    return (schedule.clock.id, 0);
                }
                if let Some(scale) = schedule.effective_tick_rate_override {
                    let original = schedule.clock.time_scale;
                    schedule.clock.time_scale = scale;
                    let _ = schedule.clock.accumulate_engine_step(engine_dt);
                    schedule.clock.time_scale = original;
                } else {
                    let _ = schedule.clock.accumulate_engine_step(engine_dt);
                }
                let due = schedule.clock.consume_ticks(schedule.tick_budget_per_step);
                (schedule.clock.id, due)
            })
            .collect()
    }

    pub fn resolve_tick_schedule(&self, work_items: &[(DimensionId, u64)]) -> TickSchedule {
        let mut ticks = Vec::new();
        let mut phases = Vec::new();
        for (dim, count) in work_items.iter() {
            let Some(schedule) = self.dimensions.iter().find(|s| s.clock.id == *dim) else {
                continue;
            };
            for offset in 0..*count {
                let tick_index = schedule.clock.local_tick + offset;
                let dt = schedule
                    .fixed_timestep
                    .unwrap_or(schedule.clock.local_tick_duration);
                ticks.push(ResolvedTick {
                    dimension: *dim,
                    tick_index,
                    priority: schedule.priority,
                    dt,
                });
                for phase in schedule.profile.phase_order.iter() {
                    phases.push(ResolvedPhase {
                        dimension: *dim,
                        tick_index,
                        priority: schedule.priority,
                        dt,
                        phase,
                    });
                }
            }
        }
        TickSchedule {
            engine_tick: self.engine_clock.tick,
            ticks,
            phases,
        }
    }
}

/// Phase hash mode placeholder for SDK-only simulation loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseHashMode {
    Off,
    PerPhase,
}

/// Minimal scheduler that runs SDK systems per phase.
pub struct Scheduler {
    phases: [Vec<Box<dyn System>>; 4],
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            phases: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        }
    }

    pub fn add_system_with_phase<S: System + 'static>(&mut self, phase: TickPhase, system: S) {
        self.phases[phase.as_index()].push(Box::new(system));
    }

    pub fn run_phase_with_timing(
        &mut self,
        world: &mut dyn RuntimeWorld,
        dt: f64,
        dimension: DimensionId,
        engine_tick: u64,
        dimension_tick: u64,
        phase: TickPhase,
    ) {
        let mut ctx = SystemRunContext::new(dimension, engine_tick, dimension_tick);
        for system in &mut self.phases[phase.as_index()] {
            system.run(world, dt, &mut ctx);
        }
    }
}

/// Deterministic driver wiring SimulationScheduler to per-dimension schedulers.
pub struct SimulationLoop {
    pub simulation_scheduler: SimulationScheduler,
    default_scheduler: Option<Scheduler>,
    per_dimension_systems: HashMap<DimensionId, Scheduler>,
    phase_hash_mode: PhaseHashMode,
}

impl SimulationLoop {
    pub fn with_default_scheduler(simulation_scheduler: SimulationScheduler, default_scheduler: Scheduler) -> Self {
        Self {
            simulation_scheduler,
            default_scheduler: Some(default_scheduler),
            per_dimension_systems: HashMap::new(),
            phase_hash_mode: PhaseHashMode::Off,
        }
    }

    pub fn register_dimension_scheduler(&mut self, dimension: DimensionId, scheduler: Scheduler) {
        self.per_dimension_systems.insert(dimension, scheduler);
    }

    pub fn step(&mut self, world: &mut dyn RuntimeWorld) -> TickSchedule {
        let work = self.simulation_scheduler.step();
        self.run_tick_schedule(&work, world)
    }

    pub fn run_tick_schedule(&mut self, work: &[(DimensionId, u64)], world: &mut dyn RuntimeWorld) -> TickSchedule {
        let schedule = self.simulation_scheduler.resolve_tick_schedule(work);
        for phase in &schedule.phases {
            if let Some(scheduler) = self.per_dimension_systems.get_mut(&phase.dimension) {
                scheduler.run_phase_with_timing(
                    world,
                    phase.dt.as_secs_f64(),
                    phase.dimension,
                    schedule.engine_tick,
                    phase.tick_index,
                    phase.phase,
                );
            } else if let Some(default) = self.default_scheduler.as_mut() {
                default.run_phase_with_timing(
                    world,
                    phase.dt.as_secs_f64(),
                    phase.dimension,
                    schedule.engine_tick,
                    phase.tick_index,
                    phase.phase,
                );
            }
        }
        schedule
    }

    pub fn set_phase_hash_mode(&mut self, mode: PhaseHashMode) {
        self.phase_hash_mode = mode;
    }
}
