"""Validation experiments for REINFORCE on ALE/Pong-v5. Uses the checkpoints written by pong.py.

Usage:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python part3_pg/validate_pong.py [--runs pong_nobase pong_value]
"""
import argparse
import glob
import os
import re
import sys

import ale_py
import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pong import (ACTION_LEFT, ACTION_RIGHT, MLP, RESULTS, discounted_returns, make_envs,  # noqa: E402
                  play_batch, preprocess, preprocess_batch)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

gym.register_envs(ale_py)
VOUT = os.path.join(RESULTS, "pong_validation")
os.makedirs(VOUT, exist_ok=True)
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results = []


def check(name, ok, evidence):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {evidence}", flush=True)


def load(run, ckpt="ckpt.pt", value=False):
    ck = torch.load(os.path.join(RESULTS, run, ckpt), map_location=DEV, weights_only=False)
    net = MLP().to(DEV)
    net.load_state_dict(ck["value" if value else "policy"])
    return net, ck["episode"]


def objects(frame80):
    """Locate our paddle (right, cols 70-71), the opponent (cols 8-9) and the ball in an 80x80 frame."""
    me = np.flatnonzero(frame80[:, 70:72].any(1))
    ball = np.argwhere(frame80[:, 10:70] > 0)
    return (me.mean() if len(me) else None), (ball[:, 0].mean() if len(ball) else None)


def test_preprocessing():
    """V1: (a) vectorised preprocessing == the provided function, (b) the action mapping really is
    RIGHT=up / LEFT=down for our (right-hand) paddle, (c) diff frame at t=0 is well defined."""
    env = gym.make("ALE/Pong-v5")
    obs, _ = env.reset(seed=0)
    frames = []
    for t in range(200):
        obs, *_ = env.step(np.random.choice([ACTION_RIGHT, ACTION_LEFT])); frames.append(obs.copy())
    F = np.stack(frames)
    same = np.array_equal(np.stack([preprocess(f.copy()).ravel() for f in F]), preprocess_batch(F))
    ys = {}
    for a in (ACTION_RIGHT, ACTION_LEFT):
        env.reset(seed=1)
        for _ in range(15):
            obs, *_ = env.step(a)
        ys[a] = objects(preprocess(obs.copy()))[0]
    check("V1 preprocessing identical to reference; RIGHT moves paddle up, LEFT down",
          same and ys[ACTION_RIGHT] < ys[ACTION_LEFT],
          f"vectorised==reference: {same}; paddle row after 15x RIGHT = {ys[ACTION_RIGHT]:.1f}, "
          f"after 15x LEFT = {ys[ACTION_LEFT]:.1f} (row 0 = top); first diff frame = frame - zeros")


def test_returns():
    """V2: returns are reset when a point is scored, so an action is only credited for the rally it
    belongs to; within a rally G_t = gamma^(k) * r_end."""
    r = np.array([0, 0, 0, -1, 0, 0, 1], dtype=float)
    G = discounted_returns(r, 0.99)
    expect = np.array([-0.99 ** 3, -0.99 ** 2, -0.99, -1, 0.99 ** 2, 0.99, 1])
    check("V2 discounted returns reset at each point", np.allclose(G, expect),
          f"r={r.tolist()} -> G={np.round(G, 4).tolist()}")


def test_progress(run, n_eps=32):
    """V3: evaluate saved snapshots over training on fresh seeds. A learning agent's score should rise
    from the random-policy level (~-20.5) roughly monotonically."""
    snaps = sorted(glob.glob(os.path.join(RESULTS, run, "ckpt_ep*.pt")),
                   key=lambda f: int(re.search(r"ep(\d+)", f).group(1)))
    if len(snaps) < 2:
        print(f"[SKIP] V3 {run}: not enough snapshots"); return
    idx = np.unique(np.linspace(0, len(snaps) - 1, min(8, len(snaps))).astype(int))
    envs = make_envs(16)
    xs, ys, es = [], [], []
    for i in idx:
        net, ep = load(run, os.path.basename(snaps[i]))
        r = [sum(e["r"]) for k in range(n_eps // 16) for e in play_batch(envs, net, DEV, seed=777 + k * 16)]
        xs.append(ep); ys.append(np.mean(r)); es.append(np.std(r) / np.sqrt(len(r)))
    envs.close()
    rho = np.corrcoef(xs, ys)[0, 1]
    check(f"V3 {run}: evaluation score improves over training", ys[-1] > ys[0] + 2 and rho > 0.7,
          ", ".join(f"ep{x}: {y:.1f}±{e:.1f}" for x, y, e in zip(xs, ys, es)) + f"; corr(ep, score)={rho:.2f}")
    return xs, ys, es


def test_tracks_ball(run, steps=4000):
    """V4: behavioural check -- a policy that has learned anything must move its paddle toward the ball:
    P(up) should be high when the ball is above the paddle and low when below."""
    net, ep = load(run)
    env = gym.make("ALE/Pong-v5")
    obs, _ = env.reset(seed=3)
    prev = np.zeros(6400, np.float32)
    rel, pup, trace = [], [], []
    for t in range(steps):
        cur = preprocess(obs.copy()).ravel()
        x = cur - prev; prev = cur
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(x).to(DEV))).item()
        me, ball = objects(cur.reshape(80, 80))
        if me is not None and ball is not None:
            rel.append(ball - me); pup.append(p)
        a = ACTION_RIGHT if np.random.rand() < p else ACTION_LEFT
        obs, r, te, tr, _ = env.step(a)
        if t < 10:
            trace.append(f"t{t}: paddle={me} ball={ball} P(up)={p:.3f} a={'UP' if a == ACTION_RIGHT else 'DN'} r={r:.0f}")
        if te or tr:
            obs, _ = env.reset(); prev[:] = 0
    rel, pup = np.array(rel), np.array(pup)
    above, below = pup[rel < -4].mean(), pup[rel > 4].mean()
    check(f"V4 {run} (ep {ep}): paddle moves toward the ball", above - below > 0.2,
          f"P(up | ball above paddle) = {above:.3f}, P(up | ball below paddle) = {below:.3f}, n={len(rel)}\n       "
          + "\n       ".join(trace))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.arange(-40, 41, 4)
    which = np.digitize(rel, bins)
    ax.plot([(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))],
            [pup[which == i].mean() if (which == i).any() else np.nan for i in range(1, len(bins))], "o-")
    ax.axhline(0.5, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("ball row − paddle row  (negative = ball above)"); ax.set_ylabel("P(move up)")
    ax.set_title(f"{run}: policy response to ball position"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(VOUT, f"{run}_tracks_ball.png"), dpi=200); plt.close(fig)


def test_gradient_variance(policy_run="pong_value", n_eps=48):
    """V5: at a frozen policy, compare the per-episode gradient with and without the learned V(s)
    baseline (same episodes): the baseline must keep the mean direction and reduce the variance."""
    net, ep = load(policy_run)
    vnet, _ = load(policy_run, value=True)
    envs = make_envs(16)
    eps = [e for k in range(n_eps // 16) for e in play_batch(envs, net, DEV, seed=9000 + 16 * k)]
    envs.close()
    g = {"none": [], "value": []}
    for e in eps:
        X = torch.from_numpy(np.stack(e["x"])).to(DEV).float()
        A = torch.tensor(np.array(e["a"]), device=DEV).float()
        G = torch.from_numpy(discounted_returns(e["r"], 0.99)).to(DEV)
        with torch.no_grad():
            V = vnet(X)
        for k, adv in (("none", G), ("value", G - V)):
            net.zero_grad()
            logp = -torch.nn.functional.binary_cross_entropy_with_logits(net(X), A, reduction="none")
            (-(logp * adv).sum()).backward()
            g[k].append(torch.cat([p.grad.flatten() for p in net.parameters()]).cpu())
    s = {k: (torch.stack(v).mean(0), torch.stack(v).var(0).sum().item()) for k, v in g.items()}
    cos = torch.nn.functional.cosine_similarity(s["none"][0], s["value"][0], dim=0).item()
    check(f"V5 Pong gradient variance with V(s) baseline lower (policy ep {ep})", s["value"][1] < s["none"][1],
          f"total variance none={s['none'][1]:.3g}, value={s['value'][1]:.3g} (x{s['none'][1]/s['value'][1]:.2f} lower); "
          f"cos(mean grads)={cos:.3f} over {len(eps)} episodes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["pong_nobase", "pong_value"])
    args = ap.parse_args()
    test_preprocessing()
    test_returns()
    prog = {}
    for run in args.runs:
        prog[run] = test_progress(run)
        test_tracks_ball(run)
    if "pong_value" in args.runs:
        test_gradient_variance()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for run, v in prog.items():
        if v:
            ax.errorbar(v[0], v[1], yerr=v[2], marker="o", capsize=3, label=run)
    ax.set_xlabel("Training episode of snapshot"); ax.set_ylabel("Mean eval reward (32 eps)")
    ax.set_title("Pong: held-out evaluation of training snapshots"); ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(VOUT, "snapshot_progress.png"), dpi=200); plt.close(fig)
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed; plots in {VOUT}")
