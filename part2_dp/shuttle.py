"""Shuttle dispatching MDP solved by enumeration (finite horizon), value iteration and policy iteration.

MDP (costs are minimised; V(s) = expected discounted cost-to-go):
  state   s in {0..S_max}            number of customers waiting at the start of the period
  action  a in {0 = wait, 1 = dispatch}
  cost    c(s,a) = cf*a + ch*max(s - a*K, 0)     (holding cost charged on customers left after dispatch)
  next    s' = min(max(s - a*K, 0) + A, S_max),  A ~ Unif{1..5}  (arrivals beyond S_max are turned away)

Usage:  python part2_dp/shuttle.py        -> solves the base problem and writes plots to results/dp/
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "dp")


class ShuttleMDP:
    def __init__(self, K=15, cf=100.0, ch=2.0, arrivals=(1, 2, 3, 4, 5), arrival_probs=None,
                 s_max=200, gamma=0.95):
        self.K, self.cf, self.ch, self.s_max, self.gamma = K, cf, ch, s_max, gamma
        self.arrivals = np.asarray(arrivals)
        self.p_arr = (np.full(len(arrivals), 1 / len(arrivals)) if arrival_probs is None
                      else np.asarray(arrival_probs, dtype=float))
        assert np.isclose(self.p_arr.sum(), 1.0)
        self.S = np.arange(s_max + 1)
        self.nS, self.nA = s_max + 1, 2
        self.C, self.P = self._build()

    def _build(self):
        C = np.zeros((self.nS, self.nA))
        P = np.zeros((self.nA, self.nS, self.nS))
        for a in range(self.nA):
            left = np.maximum(self.S - a * self.K, 0)
            C[:, a] = self.cf * a + self.ch * left
            for A, p in zip(self.arrivals, self.p_arr):
                nxt = np.minimum(left + A, self.s_max)
                np.add.at(P[a], (self.S, nxt), p)
        assert np.allclose(P.sum(axis=2), 1.0)
        return C, P

    def q_values(self, V):
        """Q(s,a) = c(s,a) + gamma * sum_s' P(s'|s,a) V(s')"""
        return self.C + self.gamma * np.stack([self.P[a] @ V for a in range(self.nA)], axis=1)


def greedy(Q, prefer=None):
    """argmin over actions; ties are broken toward `prefer` (the incumbent policy) and otherwise
    toward waiting, so that policy iteration cannot cycle between equally good policies."""
    pi = np.argmin(Q, axis=1)
    tie = np.isclose(Q[:, 0], Q[:, 1], rtol=0, atol=1e-9)
    pi[tie] = 0 if prefer is None else prefer[tie]
    return pi


def enumeration(mdp, T=500):
    """Backward induction over the finite horizon t = T-1..0 with terminal cost V_T = 0.
    Returns V[t, s] for t=0..T and the time-dependent policy pi[t, s]."""
    V = np.zeros((T + 1, mdp.nS))
    pi = np.zeros((T, mdp.nS), dtype=int)
    for t in reversed(range(T)):
        Q = mdp.q_values(V[t + 1])
        pi[t] = greedy(Q)
        V[t] = Q[np.arange(mdp.nS), pi[t]]
    return V, pi


def value_iteration(mdp, eps=1e-6, max_iter=100000, V0=None):
    """Iterate V <- min_a Q(V) until ||V_{k+1}-V_k||_inf < eps(1-gamma)/(2gamma),
    which guarantees the greedy policy is eps-optimal. Returns V, pi, history of sup-norm deltas."""
    V = np.zeros(mdp.nS) if V0 is None else V0.astype(float).copy()
    tol = eps * (1 - mdp.gamma) / (2 * mdp.gamma) if mdp.gamma > 0 else eps
    deltas = []
    for _ in range(max_iter):
        V_new = mdp.q_values(V).min(axis=1)
        deltas.append(np.max(np.abs(V_new - V)))
        V = V_new
        if deltas[-1] < tol:
            break
    return V, greedy(mdp.q_values(V)), np.array(deltas)


def policy_evaluation(mdp, pi):
    """Exact evaluation: solve (I - gamma P_pi) V = c_pi."""
    P_pi = mdp.P[pi, mdp.S, :]
    c_pi = mdp.C[mdp.S, pi]
    return np.linalg.solve(np.eye(mdp.nS) - mdp.gamma * P_pi, c_pi)


def policy_iteration(mdp, pi0=None, max_iter=1000):
    pi = np.zeros(mdp.nS, dtype=int) if pi0 is None else pi0.copy()
    history = []
    for _ in range(max_iter):
        V = policy_evaluation(mdp, pi)
        history.append((pi.copy(), V.copy()))
        pi_new = greedy(mdp.q_values(V), prefer=pi)
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    return V, pi, history


def threshold(pi):
    """Smallest s at which the policy dispatches, or None if it never does."""
    idx = np.flatnonzero(pi == 1)
    return int(idx[0]) if len(idx) else None


def is_threshold_policy(pi):
    """True if pi = 1{s >= s*} (monotone in s)."""
    s_star = threshold(pi)
    return s_star is None or (np.all(pi[:s_star] == 0) and np.all(pi[s_star:] == 1))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    mdp = ShuttleMDP()

    V_enum, pi_enum = enumeration(mdp, T=500)
    V_vi, pi_vi, deltas = value_iteration(mdp)
    V_pi, pi_pi, hist = policy_iteration(mdp)

    print(f"enumeration (T=500): dispatch threshold at t=0: s >= {threshold(pi_enum[0])}")
    print(f"value iteration: {len(deltas)} iterations, threshold s >= {threshold(pi_vi)}")
    print(f"policy iteration: {len(hist)} evaluations, threshold s >= {threshold(pi_pi)}")
    print(f"max |V_enum[0] - V_vi| = {np.max(np.abs(V_enum[0] - V_vi)):.3e}")
    print(f"max |V_vi - V_pi|      = {np.max(np.abs(V_vi - V_pi)):.3e}")
    for s in (0, 10, 50, 100, 200):
        print(f"  V*({s:3d}) = {V_pi[s]:9.3f}")

    def vplot(V, pi, title, fname):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(mdp.S, V, color="#2b6cb0", lw=2)
        s_star = threshold(pi)
        if s_star is not None:
            ax.axvline(s_star, color="#c05621", ls="--", lw=1.2, label=f"dispatch threshold s* = {s_star}")
            ax.legend(frameon=False)
        ax.set_xlabel("Number of customers waiting, s")
        ax.set_ylabel("Optimal expected discounted cost V(s)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=200)
        plt.close(fig)

    vplot(V_enum[0], pi_enum[0], "(a) Enumeration, T = 500: optimal value at t = 0", "a_enumeration_V0.png")
    vplot(V_vi, pi_vi, "(b) Value iteration (T = ∞): optimal value function", "b_value_iteration_V.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.step(mdp.S, pi_pi, where="mid", color="#2f855a", lw=2)
    ax.set_yticks([0, 1], ["0: wait", "1: dispatch"])
    ax.set_xlabel("Number of customers waiting, s")
    ax.set_ylabel("Optimal action π*(s)")
    ax.set_title(f"(c) Policy iteration: optimal policy (dispatch iff s ≥ {threshold(pi_pi)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "c_policy_iteration_pi.png"), dpi=200)
    plt.close(fig)

    np.savez(os.path.join(OUT, "solutions.npz"), V_enum0=V_enum[0], pi_enum0=pi_enum[0],
             V_vi=V_vi, pi_vi=pi_vi, V_pi=V_pi, pi_pi=pi_pi, vi_deltas=deltas)
    print(f"plots written to {OUT}")


if __name__ == "__main__":
    main()
