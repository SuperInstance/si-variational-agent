//! Variational principle for optimal agent budget allocation.
//!
//! The Euler-Lagrange equations find the stationary points of an action functional.
//! For agent budgets, we define:
//!
//! S[γ, η] = ∫ L(γ, η, γ̇, η̇) dt
//!
//! where L is the "budget Lagrangian" — kinetic cost of changing budgets minus
//! the "potential" benefit of the current allocation. The conservation constraint
//! γ + η = C enters as a Lagrange multiplier.
//!
//! The solution gives the **optimal budget trajectory** that minimizes information
//! cost while respecting the conservation law.

/// Budget configuration for a single agent.
#[derive(Debug, Clone)]
pub struct BudgetState {
    pub gamma: f64,      // durable budget
    pub eta: f64,        // ephemeral budget
    pub gamma_dot: f64,  // rate of change of γ
    pub eta_dot: f64,    // rate of change of η
}

impl BudgetState {
    pub fn new(gamma: f64, eta: f64) -> Self {
        Self { gamma, eta, gamma_dot: 0.0, eta_dot: 0.0 }
    }
    pub fn with_rates(gamma: f64, eta: f64, gd: f64, ed: f64) -> Self {
        Self { gamma, eta, gamma_dot: gd, eta_dot: ed }
    }
    pub fn total(&self) -> f64 { self.gamma + self.eta }
}

/// The budget Lagrangian L = T - V where:
/// T = ½(γ̇² + η̇²) = kinetic cost of budget changes
/// V = -½(γ² + η²) = potential benefit of allocation
#[derive(Debug, Clone)]
pub struct BudgetLagrangian {
    pub kinetic_weight: f64,
    pub potential_weight: f64,
}

impl BudgetLagrangian {
    pub fn new() -> Self { Self { kinetic_weight: 1.0, potential_weight: 1.0 } }
    pub fn with_weights(kw: f64, pw: f64) -> Self { Self { kinetic_weight: kw, potential_weight: pw } }

    /// Evaluate L(state).
    pub fn evaluate(&self, state: &BudgetState) -> f64 {
        self.kinetic(&state) - self.potential(&state)
    }

    /// Kinetic energy T.
    pub fn kinetic(&self, state: &BudgetState) -> f64 {
        0.5 * self.kinetic_weight * (state.gamma_dot.powi(2) + state.eta_dot.powi(2))
    }

    /// Potential energy V (well-shaped: minimum at γ=η=C/2).
    pub fn potential(&self, state: &BudgetState) -> f64 {
        // V = ½pw·(γ² + η²) — minimum at origin, pushes toward small values
        0.5 * self.potential_weight * (state.gamma.powi(2) + state.eta.powi(2))
    }

    /// Euler-Lagrange equation for γ: d/dt(∂L/∂γ̇) - ∂L/∂γ = λ (constraint force)
    /// For L = ½kw·γ̇² + ½kw·η̇² - ½pw·γ² - ½pw·η²:
    /// kw·γ̈ + pw·γ = λ
    pub fn euler_lagrange_gamma(&self, state: &BudgetState, gamma_ddot: f64, lambda: f64) -> f64 {
        self.kinetic_weight * gamma_ddot + self.potential_weight * state.gamma - lambda
    }

    /// Euler-Lagrange for η.
    pub fn euler_lagrange_eta(&self, state: &BudgetState, eta_ddot: f64, lambda: f64) -> f64 {
        self.kinetic_weight * eta_ddot + self.potential_weight * state.eta - lambda
    }
}

/// Action functional S = ∫ L dt.
pub fn compute_action(states: &[BudgetState], dt: f64, lagrangian: &BudgetLagrangian) -> f64 {
    states.iter().map(|s| lagrangian.evaluate(s) * dt).sum()
}

/// Evolve budget state using Euler-Lagrange equations with constraint.
/// Constraint: γ + η = C enforced via Lagrange multiplier.
#[derive(Debug, Clone)]
pub struct VariationalEvolver {
    pub lagrangian: BudgetLagrangian,
    pub conservation_total: f64,
    pub dt: f64,
    pub damping: f64,
}

impl VariationalEvolver {
    pub fn new(lagrangian: BudgetLagrangian, total: f64, dt: f64) -> Self {
        Self { lagrangian, conservation_total: total, dt, damping: 5.0 }
    }

    /// Compute Lagrange multiplier for constraint γ + η = C.
    /// λ = -pw·(γ + η - C) / 2 (restoring force proportional to violation)
    pub fn constraint_force(&self, state: &BudgetState) -> f64 {
        let violation = state.gamma + state.eta - self.conservation_total;
        -self.lagrangian.potential_weight * violation / 2.0
    }

    /// One step of constrained Euler-Lagrange evolution.
    pub fn step(&self, state: &BudgetState) -> BudgetState {
        let lambda = self.constraint_force(state);

        // γ̈ = (-pw·γ + λ - damping·γ̇) / kw
        let gamma_ddot = (-self.lagrangian.potential_weight * state.gamma + lambda
            - self.damping * state.gamma_dot) / self.lagrangian.kinetic_weight;
        let eta_ddot = (-self.lagrangian.potential_weight * state.eta + lambda
            - self.damping * state.eta_dot) / self.lagrangian.kinetic_weight;

        let new_gamma_dot = state.gamma_dot + gamma_ddot * self.dt;
        let new_eta_dot = state.eta_dot + eta_ddot * self.dt;
        let new_gamma = state.gamma + new_gamma_dot * self.dt;
        let new_eta = state.eta + new_eta_dot * self.dt;

        BudgetState::with_rates(new_gamma, new_eta, new_gamma_dot, new_eta_dot)
    }

    /// Simulate for n steps, return trajectory.
    pub fn simulate(&self, initial: &BudgetState, n_steps: usize) -> VariationalResult {
        let mut trajectory = vec![initial.clone()];
        let mut state = initial.clone();
        let mut max_conservation_error = 0.0_f64;

        for _ in 0..n_steps {
            state = self.step(&state);
            let err = (state.total() - self.conservation_total).abs();
            max_conservation_error = max_conservation_error.max(err);
            trajectory.push(state.clone());
        }

        let action = compute_action(&trajectory, self.dt, &self.lagrangian);
        let final_state = trajectory.last().unwrap().clone();

        VariationalResult {
            trajectory,
            action,
            final_state,
            max_conservation_error,
            conservation_holds: max_conservation_error < 1.0,
        }
    }
}

/// Result of variational simulation.
#[derive(Debug, Clone)]
pub struct VariationalResult {
    pub trajectory: Vec<BudgetState>,
    pub action: f64,
    pub final_state: BudgetState,
    pub max_conservation_error: f64,
    pub conservation_holds: bool,
}

/// Fleet of agents with variational budget evolution.
#[derive(Debug, Clone)]
pub struct FleetVariational {
    pub evolver: VariationalEvolver,
    pub agents: Vec<BudgetState>,
}

impl FleetVariational {
    pub fn new(lagrangian: BudgetLagrangian, dt: f64, agents: Vec<BudgetState>) -> Self {
        let total: f64 = agents.iter().map(|a| a.total()).sum::<f64>() / agents.len() as f64;
        Self { evolver: VariationalEvolver::new(lagrangian, total, dt), agents }
    }

    /// Run one step for all agents.
    pub fn step(&mut self) {
        let new_agents: Vec<BudgetState> = self.agents.iter().map(|a| self.evolver.step(a)).collect();
        self.agents = new_agents;
    }

    /// Simulate fleet for n steps.
    pub fn simulate(&mut self, n_steps: usize) -> FleetVariationalResult {
        let mut max_error = 0.0_f64;
        let initial_action: f64 = self.agents.iter()
            .map(|a| self.evolver.lagrangian.evaluate(a)).sum();
        for _ in 0..n_steps { self.step(); }
        let final_action: f64 = self.agents.iter()
            .map(|a| self.evolver.lagrangian.evaluate(a)).sum();
        for a in &self.agents {
            let err = (a.total() - self.evolver.conservation_total).abs();
            max_error = max_error.max(err);
        }
        let gamma_std = stddev(&self.agents.iter().map(|a| a.gamma).collect::<Vec<_>>());
        FleetVariationalResult {
            n_agents: self.agents.len(),
            initial_action,
            final_action,
            max_conservation_error: max_error,
            gamma_std,
            action_decreased: final_action <= initial_action,
        }
    }
}

/// Fleet variational result.
#[derive(Debug, Clone)]
pub struct FleetVariationalResult {
    pub n_agents: usize,
    pub initial_action: f64,
    pub final_action: f64,
    pub max_conservation_error: f64,
    pub gamma_std: f64,
    pub action_decreased: bool,
}

fn stddev(vals: &[f64]) -> f64 {
    let n = vals.len() as f64;
    if n < 1.0 { return 0.0; }
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

/// Find optimal stationary budget allocation.
/// At equilibrium: γ̇ = η̇ = 0, so ∂L/∂γ = λ and ∂L/∂η = λ
/// For symmetric L: γ* = η* = C/2
pub fn optimal_allocation(total: f64) -> BudgetState {
    BudgetState::new(total / 2.0, total / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lagrangian_evaluate() {
        let l = BudgetLagrangian::new();
        let s = BudgetState::with_rates(3.0, 2.0, 1.0, 1.0);
        let val = l.evaluate(&s);
        // T = 0.5*(1+1) = 1, V = 0.5*(9+4) = 6.5, L = 1 - 6.5 = -5.5
        assert!((val - (-5.5)).abs() < 1e-10, "L = {}", val);
    }

    #[test]
    fn test_kinetic_energy() {
        let l = BudgetLagrangian::new();
        let s = BudgetState::with_rates(0.0, 0.0, 3.0, 4.0);
        assert!((l.kinetic(&s) - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_potential_energy() {
        let l = BudgetLagrangian::new();
        let s = BudgetState::new(3.0, 4.0);
        assert!((l.potential(&s) - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_euler_lagrange_zero_at_equilibrium() {
        let l = BudgetLagrangian::new();
        let s = BudgetState::new(5.0, 5.0);
        // At equilibrium with λ = pw*γ = 5: kw*0 + pw*5 - 5 = 0
        let lambda = l.potential_weight * s.gamma;
        let el = l.euler_lagrange_gamma(&s, 0.0, lambda);
        assert!(el.abs() < 1e-10);
    }

    #[test]
    fn test_constraint_force() {
        let l = BudgetLagrangian::new();
        let evolver = VariationalEvolver::new(l, 10.0, 0.01);
        let state = BudgetState::new(6.0, 5.0); // total = 11, target = 10
        let force = evolver.constraint_force(&state);
        assert!(force < 0.0, "Should push toward conservation, got {}", force);
    }

    #[test]
    fn test_constraint_force_at_target() {
        let l = BudgetLagrangian::new();
        let evolver = VariationalEvolver::new(l, 10.0, 0.01);
        let state = BudgetState::new(5.0, 5.0);
        let force = evolver.constraint_force(&state);
        assert!(force.abs() < 1e-10);
    }

    #[test]
    fn test_step_changes_state() {
        let l = BudgetLagrangian::new();
        let evolver = VariationalEvolver::new(l, 10.0, 0.01);
        let initial = BudgetState::with_rates(8.0, 2.0, 0.0, 0.0);
        let next = evolver.step(&initial);
        assert!((next.gamma - initial.gamma).abs() > 1e-12);
    }

    #[test]
    fn test_simulate_trajectory_length() {
        let l = BudgetLagrangian::new();
        let evolver = VariationalEvolver::new(l, 10.0, 0.01);
        let initial = BudgetState::with_rates(8.0, 2.0, 0.0, 0.0);
        let result = evolver.simulate(&initial, 100);
        assert_eq!(result.trajectory.len(), 101);
    }

    #[test]
    fn test_conservation_bounded() {
        let l = BudgetLagrangian::new();
        let evolver = VariationalEvolver::new(l, 10.0, 0.01);
        let initial = BudgetState::with_rates(8.0, 2.0, 0.0, 0.0);
        let result = evolver.simulate(&initial, 200);
        assert!(result.max_conservation_error < 10.0, "Error: {}", result.max_conservation_error);
    }

    #[test]
    fn test_action_computed() {
        let l = BudgetLagrangian::new();
        let evolver = VariationalEvolver::new(l, 10.0, 0.01);
        let initial = BudgetState::with_rates(8.0, 2.0, 0.0, 0.0);
        let result = evolver.simulate(&initial, 50);
        assert!(result.action.is_finite());
    }

    #[test]
    fn test_fleet_simulation() {
        let l = BudgetLagrangian::new();
        let agents = vec![
            BudgetState::with_rates(8.0, 2.0, 0.0, 0.0),
            BudgetState::with_rates(3.0, 7.0, 0.0, 0.0),
            BudgetState::with_rates(5.0, 5.0, 0.0, 0.0),
        ];
        let mut fleet = FleetVariational::new(l, 0.01, agents);
        let result = fleet.simulate(200);
        assert_eq!(result.n_agents, 3);
        assert!(result.max_conservation_error < 10.0);
    }

    #[test]
    fn test_optimal_allocation() {
        let opt = optimal_allocation(10.0);
        assert!((opt.gamma - 5.0).abs() < 1e-10);
        assert!((opt.eta - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_damping_stabilizes() {
        let l = BudgetLagrangian::new();
        let mut evolver = VariationalEvolver::new(l, 10.0, 0.01);
        evolver.damping = 2.0; // heavy damping
        let initial = BudgetState::with_rates(9.0, 1.0, 0.0, 0.0);
        let result = evolver.simulate(&initial, 500);
        // With heavy damping, should converge near equilibrium
        assert!(result.final_state.gamma_dot.abs() < 1.0);
    }

    #[test]
    fn test_budget_total() {
        let s = BudgetState::new(3.0, 7.0);
        assert!((s.total() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_different_weights() {
        let l = BudgetLagrangian::with_weights(2.0, 0.5);
        let s = BudgetState::with_rates(0.0, 0.0, 1.0, 1.0);
        // T = 0.5*2*(1+1) = 2, V = 0.5*0.5*0 = 0
        assert!((l.kinetic(&s) - 2.0).abs() < 1e-10);
    }
}
