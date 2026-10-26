"""Validation experiments for the shuttle-dispatch solvers. Each test prints a PASS/FAIL line together
with the quantitative evidence; plots are written to results/dp/validation/.

Usage:  python part2_dp/validate_dp.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")  # small matrices; don't steal cores from Pong training
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shuttle import (OUT, ShuttleMDP, enumeration, is_threshold_policy, policy_evaluation,  # noqa: E402
                     policy_iteration, threshold, value_iteration)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VOUT = os.path.join(OUT, "validation")
os.makedirs(VOUT, exist_ok=True)
results = []


def check(name, ok, evidence):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {evidence}")


def test_solvers_agree():
    """T1: three independent algorithms must give the same answer. Enumeration with T=500 differs from
    the infinite-horizon value by at most gamma^T * max|V*| (tail cost after T)."""
    mdp = ShuttleMDP()
    V_e, pi_e = enumeration(mdp, 500)
    V_v, pi_v, _ = value_iteration(mdp, eps=1e-8)
    V_p, pi_p, _ = policy_iteration(mdp)
    bound = mdp.gamma ** 500 * V_p.max()
    d_ev, d_vp = np.max(np.abs(V_e[0] - V_p)), np.max(np.abs(V_v - V_p))
    check("T1 enumeration / VI / PI agree",
          d_ev <= bound + 1e-6 and d_vp < 1e-6 and np.array_equal(pi_e[0], pi_p) and np.array_equal(pi_v, pi_p),
          f"|V_enum0-V_PI|={d_ev:.2e} (bound gamma^T*maxV={bound:.2e}), |V_VI-V_PI|={d_vp:.2e}, "
          f"policies identical={np.array_equal(pi_e[0], pi_p) and np.array_equal(pi_v, pi_p)}")


def test_bellman_optimality():
    """T2: V* must be a fixed point of the Bellman optimality operator, and no single-state deviation
    from pi* can lower the cost (one-step deviation check)."""
    mdp = ShuttleMDP()
    V, pi, _ = policy_iteration(mdp)
    resid = np.max(np.abs(mdp.q_values(V).min(axis=1) - V))
    worst = np.inf
    for s in mdp.S:
        pi2 = pi.copy(); pi2[s] = 1 - pi2[s]
        worst = min(worst, np.min(policy_evaluation(mdp, pi2) - V))
    check("T2 Bellman residual ~ 0 and no profitable one-state deviation",
          resid < 1e-8 and worst > -1e-8,
          f"||T V - V||_inf = {resid:.2e}; min over 201 flipped policies of (V_flip - V*) = {worst:.3e} (>= 0)")


def test_monte_carlo(n_paths=4000, horizon=400, seed=0):
    """T3: simulate the real stochastic system (independent of the matrices P, C) under pi* and compare
    the empirical discounted cost with V*(s0)."""
    mdp = ShuttleMDP()
    V, pi, _ = policy_iteration(mdp)
    rng = np.random.default_rng(seed)
    rows = []
    for s0 in (0, 13, 50, 120, 200):
        s = np.full(n_paths, s0)
        total = np.zeros(n_paths)
        for t in range(horizon):
            a = pi[s]
            left = np.maximum(s - a * mdp.K, 0)
            total += mdp.gamma ** t * (mdp.cf * a + mdp.ch * left)
            s = np.minimum(left + rng.integers(1, 6, n_paths), mdp.s_max)
        se = total.std(ddof=1) / np.sqrt(n_paths)
        rows.append((s0, V[s0], total.mean(), se, abs(total.mean() - V[s0]) / se))
    ok = all(z < 4 for *_, z in rows)
    ev = "; ".join(f"s0={s0}: V*={v:.1f} MC={m:.1f}±{se:.1f} (z={z:.2f})" for s0, v, m, se, z in rows)
    check("T3 Monte-Carlo simulation matches V* (|z| < 4)", ok, ev)


def test_structure():
    """T4: theory (holding cost convex & increasing) predicts V* nondecreasing in s and a threshold
    (control-limit) policy."""
    mdp = ShuttleMDP()
    V, pi, _ = policy_iteration(mdp)
    mono = np.all(np.diff(V) >= -1e-9)
    slope = np.mean(np.diff(V[150:190]))
    check("T4 V* nondecreasing, pi* is a threshold policy",
          mono and is_threshold_policy(pi),
          f"min diff V = {np.diff(V).min():.3f}, threshold s* = {threshold(pi)}, "
          f"threshold-shaped = {is_threshold_policy(pi)}, mean slope dV/ds on [150,190] = {slope:.2f}")


def test_limiting_parameters():
    """T5: parameter values whose optimal policy can be derived by hand."""
    cases = []
    # cf = 0: dispatching is free, so dispatch whenever anyone waits (s>=1); at s=0 wait (tie -> wait)
    pi = policy_iteration(ShuttleMDP(cf=0))[1]
    cases.append(("cf=0 -> dispatch iff s>=1", threshold(pi) == 1 and is_threshold_policy(pi), f"s*={threshold(pi)}"))
    # ch = 0: waiting is free, never pay cf
    pi = policy_iteration(ShuttleMDP(ch=0))[1]
    cases.append(("ch=0 -> never dispatch", threshold(pi) is None, f"s*={threshold(pi)}"))
    # cf larger than the cost of holding 200 people forever (ch*200/(1-gamma)=8000) -> never dispatch
    pi = policy_iteration(ShuttleMDP(cf=1e5))[1]
    cases.append(("cf=1e5 -> never dispatch", threshold(pi) is None, f"s*={threshold(pi)}"))
    # gamma = 0 (myopic): dispatch iff cf + ch*max(s-K,0) < ch*s  <=>  s > cf/ch (when cf/ch < K)
    pi = policy_iteration(ShuttleMDP(cf=20, gamma=0.0))[1]
    cases.append(("gamma=0, cf=20 -> dispatch iff s > cf/ch = 10", threshold(pi) == 11, f"s*={threshold(pi)}"))
    pi = policy_iteration(ShuttleMDP(gamma=0.0))[1]
    cases.append(("gamma=0, cf=100 > ch*K=30 -> never dispatch", threshold(pi) is None, f"s*={threshold(pi)}"))
    # unlimited capacity: under the K=inf policy (dispatch iff s >= s_inf) the largest reachable state is
    # (s_inf - 1) + max(A) = s_inf + 4. Capacity therefore never binds iff K >= s_inf + 4, so those K
    # must reproduce the K=inf threshold exactly, while K just below it may not.
    s_inf = threshold(policy_iteration(ShuttleMDP(K=1000))[1])
    s_K = {k: threshold(policy_iteration(ShuttleMDP(K=k))[1]) for k in (s_inf, s_inf + 3, s_inf + 4, 50, 200)}
    s15 = threshold(policy_iteration(ShuttleMDP())[1])
    cases.append((f"K=inf: s*(inf)={s_inf}; identical for every K >= s*(inf)+4, and s*(inf) >= s*(K=15)",
                  all(s_K[k] == s_inf for k in s_K if k >= s_inf + 4) and s_inf >= s15,
                  f"s*(K) = {s_K}, s*(K=15) = {s15}"))
    for name, ok, ev in cases:
        check(f"T5 {name}", ok, ev)


def test_deterministic_hand_calc():
    """T6: with deterministic arrivals A=5 and K=15 the system is periodic and can be solved by hand.
    Starting empty and dispatching every time s reaches 15 (cycle 0->5->10->15->dispatch->5...):
    compare the solver's value with the closed-form cost of the policy it returns."""
    mdp = ShuttleMDP(arrivals=(5,))
    V, pi, _ = policy_iteration(mdp)
    s, cost, trace = 0, 0.0, []
    for t in range(2000):  # explicit roll-out, no matrices
        a = pi[s]
        left = max(s - a * mdp.K, 0)
        c = mdp.cf * a + mdp.ch * left
        if t < 10:
            trace.append(f"t{t}: s={s} a={a} c={c:.0f}")
        cost += mdp.gamma ** t * c
        s = min(left + 5, mdp.s_max)
    check("T6 deterministic arrivals: roll-out cost equals V*(0)", abs(cost - V[0]) < 1e-6,
          f"V*(0)={V[0]:.4f}, roll-out={cost:.4f}, s*={threshold(pi)}\n       first 10 steps: " + ", ".join(trace))


def test_comparative_statics():
    """T7: threshold should increase with cf, decrease with ch, increase with K. Plotted."""
    cfs = np.arange(10, 401, 10)
    ths_cf = [threshold(policy_iteration(ShuttleMDP(cf=c))[1]) for c in cfs]
    chs = np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8, 10])
    ths_ch = [threshold(policy_iteration(ShuttleMDP(ch=c))[1]) for c in chs]
    Ks = np.arange(5, 41, 5)
    ths_K = [threshold(policy_iteration(ShuttleMDP(K=k))[1]) for k in Ks]
    ok = (np.all(np.diff(ths_cf) >= 0) and np.all(np.diff(ths_ch) <= 0) and np.all(np.diff(ths_K) >= 0))
    check("T7 threshold monotone in cf (up), ch (down), K (up)", ok,
          f"cf 10..400 -> s* {ths_cf[0]}..{ths_cf[-1]}; ch {chs.tolist()} -> {ths_ch}; K {Ks.tolist()} -> {ths_K}")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, x, y, lab in zip(axs, (cfs, chs, Ks), (ths_cf, ths_ch, ths_K),
                             ("Dispatch cost $c_f$", "Holding cost $c_h$", "Capacity $K$")):
        ax.plot(x, y, "o-", ms=3)
        ax.set_xlabel(lab); ax.set_ylabel("Dispatch threshold s*"); ax.grid(alpha=0.3)
    fig.suptitle("Comparative statics of the optimal threshold")
    fig.tight_layout(); fig.savefig(os.path.join(VOUT, "threshold_sensitivity.png"), dpi=200); plt.close(fig)


def test_convergence():
    """T8: VI is a gamma-contraction, so ||V_{k+1}-V_k|| should shrink by a factor -> gamma=0.95;
    the finite-horizon V_0^T approaches V* like gamma^T; PI converges in a handful of iterations,
    and from any start policy."""
    mdp = ShuttleMDP()
    V_star, pi_star, _ = policy_iteration(mdp)
    _, _, deltas = value_iteration(mdp, eps=1e-10)
    ratio = deltas[50:] / deltas[49:-1]
    Ts = np.arange(0, 501, 10)
    V_all, _ = enumeration(mdp, 500)
    gaps = [np.max(np.abs(V_all[500 - T] - V_star)) for T in Ts]  # V_all[t] is optimal for horizon 500-t
    starts = {"never": np.zeros(mdp.nS, int), "always": np.ones(mdp.nS, int),
              "random": np.random.default_rng(1).integers(0, 2, mdp.nS)}
    pi_iters = {k: len(policy_iteration(mdp, pi0=v)[2]) for k, v in starts.items()}
    pi_same = all(np.array_equal(policy_iteration(mdp, pi0=v)[1], pi_star) for v in starts.values())
    check("T8 VI contraction rate ~ gamma, PI converges from any start",
          abs(np.median(ratio) - mdp.gamma) < 0.01 and pi_same,
          f"median ||dV_k+1||/||dV_k|| = {np.median(ratio):.4f} (gamma={mdp.gamma}); "
          f"VI iters to 1e-10: {len(deltas)}; PI iterations by start: {pi_iters}; same pi*: {pi_same}")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].semilogy(deltas, label=r"$\|V_{k+1}-V_k\|_\infty$")
    axs[0].semilogy(deltas[0] * mdp.gamma ** np.arange(len(deltas)), "--", label=r"$\gamma^k$ reference")
    axs[0].set_xlabel("VI iteration k"); axs[0].legend(frameon=False); axs[0].grid(alpha=0.3)
    axs[0].set_title("Value iteration convergence")
    axs[1].semilogy(Ts, np.maximum(gaps, 1e-16), label=r"$\|V_0^{T}-V^*\|_\infty$")
    axs[1].semilogy(Ts, V_star.max() * mdp.gamma ** Ts, "--", label=r"$\gamma^T \max V^*$")
    axs[1].set_xlabel("Horizon T"); axs[1].legend(frameon=False); axs[1].grid(alpha=0.3)
    axs[1].set_title("Finite-horizon value approaches $V^*$")
    fig.tight_layout(); fig.savefig(os.path.join(VOUT, "convergence.png"), dpi=200); plt.close(fig)


def test_finite_horizon_end_effect():
    """T9 (edge case): near the end of a finite horizon, dispatching is not worth it because the saved
    holding costs stop at T. The time-dependent threshold should rise as t -> T, and at t=T-1 the agent
    dispatches only if the immediate saving ch*min(s,K) exceeds cf, i.e. never (ch*K=30 < cf=100)."""
    mdp = ShuttleMDP()
    _, pi = enumeration(mdp, 500)
    ths = [threshold(pi[t]) for t in range(500)]
    last_change = max(t for t in range(1, 500) if ths[t] != ths[t - 1])
    check("T9 finite horizon: stationary threshold early, no dispatch in last period",
          ths[0] == ths[400] and ths[-1] is None,
          f"s*(t=0)={ths[0]}, s*(t=400)={ths[400]}, s*(t=490..499)={ths[490:]}; threshold constant up to t={last_change - 1}")


def test_capacity_cap():
    """T10 (edge case): the station cap of 200 is an artificial boundary. Raising it to 400 should leave
    V* and pi* unchanged where the optimally-controlled system lives (s well below 200)."""
    V200, pi200, _ = policy_iteration(ShuttleMDP(s_max=200))
    V400, pi400, _ = policy_iteration(ShuttleMDP(s_max=400))
    d_low = np.max(np.abs(V200[:100] - V400[:100]))
    d_high = abs(V200[200] - V400[200])
    check("T10 cap at 200 does not affect the policy / low-state values",
          np.array_equal(pi200, pi400[:201]) and d_low < 1e-3,
          f"max|V200-V400| on s<100 = {d_low:.2e}; at s=200: {d_high:.2f} (boundary effect only); same policy: "
          f"{np.array_equal(pi200, pi400[:201])}")


if __name__ == "__main__":
    for f in (test_solvers_agree, test_bellman_optimality, test_monte_carlo, test_structure,
              test_limiting_parameters, test_deterministic_hand_calc, test_comparative_statics,
              test_convergence, test_finite_horizon_end_effect, test_capacity_cap):
        f()
    n_ok = sum(ok for _, ok in results)
    print(f"\n{n_ok}/{len(results)} checks passed; plots in {VOUT}")
