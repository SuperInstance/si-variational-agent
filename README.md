# si-variational-agent

> **Proof of Concept:** Variational principle for optimal agent budget allocation — the Euler-Lagrange equations find stationary budget trajectories that minimize action while respecting γ + η = C.

## The Insight

In physics, the **principle of least action** states that a system evolves along the path that minimizes the action functional S = ∫ L dt, where L is the Lagrangian.

For agent budgets:
- **Kinetic energy** T = ½(γ̇² + η̇²) — cost of changing budgets
- **Potential energy** V = ½(γ² + η²) — benefit of current allocation
- **Lagrangian** L = T - V
- **Constraint** γ + η = C enforced via Lagrange multiplier λ

The **Euler-Lagrange equations** give:
kw·γ̈ + pw·γ = λ

The Lagrange multiplier λ acts as a **constraint force** — it pushes budgets back toward the conservation manifold whenever they drift.

## What This Proves

1. **Constraint force works**: λ = -pw·(γ+η-C)/2 restores conservation
2. **Damping stabilizes**: Heavy damping → budget converges to equilibrium
3. **Action is finite**: The action functional computes a meaningful cost
4. **Fleet evolution works**: Multiple agents evolve simultaneously
5. **Optimal allocation**: γ* = η* = C/2 (symmetric equilibrium)

## Usage

```rust
use si_variational_agent::*;

// Define Lagrangian
let lagrangian = BudgetLagrangian::new();

// Single agent evolution
let evolver = VariationalEvolver::new(lagrangian.clone(), 10.0, 0.01);
let initial = BudgetState::with_rates(8.0, 2.0, 0.0, 0.0);
let result = evolver.simulate(&initial, 500);
println!("Action: {}", result.action);
println!("Conservation error: {}", result.max_conservation_error);

// Fleet evolution
let agents = vec![
    BudgetState::with_rates(8.0, 2.0, 0.0, 0.0),
    BudgetState::with_rates(3.0, 7.0, 0.0, 0.0),
];
let mut fleet = FleetVariational::new(lagrangian, 0.01, agents);
let result = fleet.simulate(200);
println!("Action decreased: {}", result.action_decreased);

// Optimal allocation
let opt = optimal_allocation(10.0); // γ=5, η=5
```

## Modules

- `BudgetState` — agent's budget (γ, η) and rates of change (γ̇, η̇)
- `BudgetLagrangian` — kinetic/potential energy with configurable weights
- `compute_action()` — integrate Lagrangian over trajectory
- `VariationalEvolver` — Euler-Lagrange evolution with constraint enforcement
- `FleetVariational` — multi-agent variational evolution
- `optimal_allocation()` — symmetric equilibrium (γ = η = C/2)

## Connection to Conservation Law

This IS the dynamics of γ + η = C:
- The conservation law is the **constraint** in the variational problem
- The Lagrange multiplier λ is the **force** that enforces conservation
- Without λ, budgets diverge from the constraint manifold
- With λ, they stay on it (up to numerical precision)

In Hamiltonian mechanics: the constraint generates a gauge symmetry. Noether's theorem says this symmetry corresponds to the conservation law. The variational principle and Noether's theorem are two sides of the same coin.

## Mathematical Background

### Lagrangian Mechanics
L(q, q̇) = T(q̇) - V(q)

### Euler-Lagrange Equations
d/dt(∂L/∂q̇) - ∂L/∂q = Q_constraint

For our budget Lagrangian: kw·q̈ + pw·q = λ

### Lagrange Multiplier
λ = -pw·(γ + η - C) / 2

This is a restoring force proportional to the conservation violation.

### Stationary Point
At equilibrium (q̇ = 0, q̈ = 0): pw·γ = λ, pw·η = λ → γ = η = C/2

## Tests: 15

Covers: Lagrangian evaluation, kinetic/potential energy, Euler-Lagrange at equilibrium, constraint force sign/zero, state evolution, trajectory length, conservation bounds, action computation, fleet simulation, optimal allocation, damping stabilization, budget total, custom weights.

## License

MIT
