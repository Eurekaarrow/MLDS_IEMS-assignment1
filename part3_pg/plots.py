"""Report figures for the policy-gradient parts.

  python part3_pg/plots.py cartpole --configs none_C_lr1e-3_scale value_C_lr1e-3_scale --seed 0
  python part3_pg/plots.py pong --runs pong_nobase pong_value
  python part3_pg/plots.py sweep            (summary table of every CartPole config/seed)
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
LABEL = {"none": "Part 1: no baseline", "mean": "Part 2: constant (mean-return) baseline",
         "value": "Part 2: learned value baseline V(s)"}


def ma(x, w=100):
    """Trailing moving average; for the first w-1 episodes it averages what is available."""
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0))
    n = np.arange(1, len(x) + 1)
    return (c[n] - c[np.maximum(n - w, 0)]) / np.minimum(n, w)


def curve(ax, ep, r, title, ylim=None):
    ax.plot(ep, r, color="#90cdf4", lw=0.6, alpha=0.8, label="Episode reward")
    ax.plot(ep, ma(r), color="#2b6cb0", lw=2, label="100-episode moving average")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Episode reward"); ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3); ax.legend(frameon=False, loc="lower right")


def hist(ax, rewards, title, bins):
    ax.hist(rewards, bins=bins, color="#48bb78", edgecolor="white")
    ax.axvline(rewards.mean(), color="#2f855a", ls="--", lw=1.5)
    ax.set_title(f"{title}\nmean = {rewards.mean():.2f}, std = {rewards.std():.2f}  (500 episodes)")
    ax.set_xlabel("Episode reward"); ax.set_ylabel("Count"); ax.grid(alpha=0.3)


def cartpole(configs, seed):
    base = os.path.join(ROOT, "results", "cartpole")
    for cfg in configs:
        d = os.path.join(base, cfg, f"seed{seed}")
        df = pd.read_csv(os.path.join(d, "episodes.csv"))
        kind = cfg.split("_")[0]
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
        curve(axs[0], df.episode, df.reward, f"CartPole-v1 training — {LABEL[kind]}", (0, 520))
        ev = os.path.join(d, "eval_stochastic.npy")
        if os.path.exists(ev):
            hist(axs[1], np.load(ev), "Trained policy, 500 roll-outs", np.arange(0, 525, 25))
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"cartpole_{cfg}_seed{seed}.png"), dpi=200); plt.close(fig)
    # all seeds, mean ± range, per config
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for cfg, col in zip(configs, ["#718096", "#3182ce", "#dd6b20", "#38a169"]):
        runs = [pd.read_csv(f).reward.values for f in sorted(glob.glob(os.path.join(base, cfg, "seed*", "episodes.csv")))]
        M = np.stack([ma(r) for r in runs])
        x = np.arange(1, M.shape[1] + 1)
        ax.plot(x, M.mean(0), color=col, lw=2, label=f"{LABEL[cfg.split('_')[0]]} ({len(runs)} seeds)")
        ax.fill_between(x, M.min(0), M.max(0), color=col, alpha=0.15)
    ax.set_xlabel("Training episode"); ax.set_ylabel("100-episode moving average reward")
    ax.set_title("CartPole-v1: mean over seeds (band = min–max)"); ax.set_ylim(0, 520)
    ax.grid(alpha=0.3); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cartpole_seeds_comparison.png"), dpi=200); plt.close(fig)
    print(f"figures in {FIG}")


def pong(runs):
    base = os.path.join(ROOT, "results")
    for run in runs:
        df = pd.read_csv(os.path.join(base, run, "episodes.csv"))
        kind = "value" if "value" in run else "none"
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
        curve(axs[0], df.episode, df.reward, f"Pong-v5 training — {LABEL[kind]}", (-21.5, 21.5))
        ev = os.path.join(base, run, "eval_stochastic.npy")
        if os.path.exists(ev):
            hist(axs[1], np.load(ev), "Trained policy, 500 roll-outs", np.arange(-21.5, 22, 1))
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{run}.png"), dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for run, col in zip(runs, ["#718096", "#3182ce", "#dd6b20", "#38a169"]):
        df = pd.read_csv(os.path.join(base, run, "episodes.csv"))
        ax.plot(df.episode, ma(df.reward), color=col, lw=2, label=run)
    ax.set_xlabel("Training episode"); ax.set_ylabel("100-episode moving average reward")
    ax.set_title("Pong-v5: baseline comparison"); ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pong_comparison.png"), dpi=200); plt.close(fig)
    print(f"figures in {FIG}")


def sweep():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "cartpole", "*", "seed*", "episodes.csv"))):
        cfg, seed = f.split(os.sep)[-3], f.split(os.sep)[-2]
        r = pd.read_csv(f).reward.values
        m = ma(r)
        first = np.argmax(m >= 475) + 1 if (m >= 475).any() else np.nan
        rows.append((cfg, seed, len(r), r[-100:].mean(), first, (r[-100:] < 20).mean()))
    df = pd.DataFrame(rows, columns=["config", "seed", "episodes", "last100_mean", "first_ep_MA>=475", "collapsed_frac"])
    pd.set_option("display.width", 200)
    print(df.groupby("config").agg(seeds=("seed", "count"), last100_mean=("last100_mean", "mean"),
                                   last100_min=("last100_mean", "min"), solved_seeds=("first_ep_MA>=475", "count"),
                                   median_ep_to_solve=("first_ep_MA>=475", "median")).round(1).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["cartpole", "pong", "sweep"])
    p.add_argument("--configs", nargs="+", default=[])
    p.add_argument("--runs", nargs="+", default=["pong_nobase", "pong_value"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"cartpole": lambda: cartpole(a.configs, a.seed), "pong": lambda: pong(a.runs), "sweep": sweep}[a.what]()
