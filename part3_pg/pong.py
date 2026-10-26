"""REINFORCE on ALE/Pong-v5 (actions restricted to RIGHT=2 / LEFT=3).

Part 1: --baseline none   (vanilla REINFORCE, returns scaled by their batch std)
Part 2: --baseline value  (learned state-value baseline V(s), advantage = G - V(s))

Training is synchronous on-policy: N parallel envs each play exactly one episode
with the current policy, then the policy is updated on that batch of N episodes.

Examples
  python part3_pg/pong.py train --baseline none  --run pong_nobase
  python part3_pg/pong.py train --baseline value --run pong_value
  python part3_pg/pong.py eval  --run pong_value --episodes 500
"""
import argparse
import csv
import os
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

gym.register_envs(ale_py)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
ACTION_RIGHT, ACTION_LEFT = 2, 3


def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float32).ravel(), [80, 80])


def preprocess_batch(obs):
    """Vectorised version of `preprocess` for obs of shape (N,210,160,3) -> (N,6400)."""
    img = obs[:, 35:195:2, ::2, 0].copy()
    img[img == 144] = 0
    img[img == 109] = 0
    return (img != 0).astype(np.float32).reshape(len(obs), 6400)


def discounted_returns(rewards, gamma):
    """G_t = r_t + gamma*G_{t+1}, reset to 0 at every point scored (r != 0):
    each Pong rally is an independent sub-game, so credit is not carried across points."""
    G = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running = 0.0
        running = rewards[t] + gamma * running
        G[t] = running
    return G


class MLP(nn.Module):
    def __init__(self, hidden=200):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6400, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_envs(n):
    def thunk():
        import ale_py as _ale
        gym.register_envs(_ale)
        return gym.make("ALE/Pong-v5")
    return gym.vector.AsyncVectorEnv([thunk for _ in range(n)])


@torch.no_grad()
def play_batch(envs, policy, device, seed=None, greedy=False):
    """Every env plays exactly one full episode under the current policy.
    Returns a list of per-episode dicts with x (int8 diff frames), a (1=RIGHT), r."""
    n = envs.num_envs
    obs, _ = envs.reset(seed=seed)
    prev = np.zeros((n, 6400), dtype=np.float32)  # first diff frame = current frame
    active = np.ones(n, dtype=bool)
    eps = [{"x": [], "a": [], "r": []} for _ in range(n)]
    while active.any():
        cur = preprocess_batch(obs)
        x = cur - prev
        prev = cur
        p_right = torch.sigmoid(policy(torch.from_numpy(x).to(device))).cpu().numpy()
        a = (p_right > 0.5) if greedy else (np.random.rand(n) < p_right)
        obs, r, term, trunc, _ = envs.step(np.where(a, ACTION_RIGHT, ACTION_LEFT))
        for i in np.flatnonzero(active):
            eps[i]["x"].append(x[i].astype(np.int8))
            eps[i]["a"].append(a[i])
            eps[i]["r"].append(r[i])
            if term[i] or trunc[i]:
                active[i] = False  # env autoresets; ignore it until the next batch
    return eps


def train(args):
    run_dir = os.path.join(RESULTS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = MLP(args.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    value = value_opt = None
    if args.baseline == "value":
        value = MLP(args.hidden).to(device)
        value_opt = torch.optim.Adam(value.parameters(), lr=args.value_lr)

    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    log_path = os.path.join(run_dir, "episodes.csv")
    upd_path = os.path.join(run_dir, "updates.csv")
    episode, update = 0, 0
    if os.path.exists(ckpt_path):  # resume
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ck["policy"]); opt.load_state_dict(ck["opt"])
        if value is not None:
            value.load_state_dict(ck["value"]); value_opt.load_state_dict(ck["value_opt"])
        episode, update = ck["episode"], ck["update"]
        np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"].cpu())
        # drop log rows written after the checkpoint so that episode numbers are not duplicated
        for path, col, last in ((log_path, "episode", episode), (upd_path, "update", update)):
            with open(path) as f:
                rows = list(csv.reader(f))
            k = rows[0].index(col)
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if int(r[k]) <= last])
        print(f"resumed from episode {episode}, update {update}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "reward", "length", "update", "wall_s"])
        with open(upd_path, "w", newline="") as f:
            csv.writer(f).writerow(["update", "episode", "mean_reward", "steps", "policy_loss",
                                    "grad_norm", "value_loss", "adv_std", "mean_p_right", "wall_s"])
        with open(os.path.join(run_dir, "config.txt"), "w") as f:
            f.write(str(vars(args)) + "\n")

    envs = make_envs(args.num_envs)
    t0 = time.time()
    first = episode == 0
    while episode < args.max_episodes and (time.time() - t0) < args.hours * 3600:
        eps = play_batch(envs, policy, device, seed=args.seed if first else None)
        first = False

        # frames stay on the CPU as int8 (diffs are in {-1,0,1}); only one chunk at a time goes to the GPU
        X = torch.from_numpy(np.concatenate([np.stack(e["x"]) for e in eps]))
        xb = lambda idx: X[idx.cpu()].to(device).float()
        A = torch.from_numpy(np.concatenate([np.array(e["a"]) for e in eps])).to(device).float()
        G = torch.from_numpy(np.concatenate([discounted_returns(e["r"], args.gamma) for e in eps])).to(device)

        value_loss = float("nan")
        if value is not None:
            # fit V(s) to observed returns, then use it (frozen) as the baseline
            for _ in range(args.value_iters):
                value_opt.zero_grad()
                vl = 0.0
                for idx in torch.split(torch.randperm(len(G), device=device), args.chunk):
                    l = ((value(xb(idx)) - G[idx]) ** 2).sum() / len(G)
                    l.backward(); vl += l.item()
                value_opt.step()
                value_loss = vl
            with torch.no_grad():
                V = torch.cat([value(xb(idx)) for idx in torch.split(torch.arange(len(G), device=device), args.chunk)])
            adv = G - V
        else:
            adv = G
        # scale only (no centering): keeps Part 1 a pure no-baseline estimator
        adv_std = adv.std().item()
        if np.isfinite(adv_std) and adv_std > 1e-6:  # guard all-equal / single-step batches
            adv = adv / adv_std

        opt.zero_grad()
        ploss, p_sum = 0.0, 0.0
        for idx in torch.split(torch.arange(len(G), device=device), args.chunk):
            logits = policy(xb(idx))
            # log pi(a|s) for Bernoulli policy: a=1 -> logsigmoid(z), a=0 -> logsigmoid(-z)
            logp = -nn.functional.binary_cross_entropy_with_logits(logits, A[idx], reduction="none")
            l = -(logp * adv[idx]).sum() / len(G)
            l.backward(); ploss += l.item()
            p_sum += torch.sigmoid(logits).sum().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf")).item()
        opt.step()
        update += 1
        del X
        torch.cuda.empty_cache()

        wall = time.time() - t0
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            for e in eps:
                episode += 1
                w.writerow([episode, float(np.sum(e["r"])), len(e["r"]), update, round(wall, 1)])
        rewards = [float(np.sum(e["r"])) for e in eps]
        with open(upd_path, "a", newline="") as f:
            csv.writer(f).writerow([update, episode, np.mean(rewards), len(G), ploss, grad_norm,
                                    value_loss, adv_std, p_sum / len(G), round(wall, 1)])
        print(f"upd {update:5d} ep {episode:6d} | mean R {np.mean(rewards):6.2f} | steps {len(G):6d} "
              f"| gnorm {grad_norm:.3f} | p_right {p_sum/len(G):.3f} | {wall/60:.1f} min", flush=True)

        if update % args.save_every == 0:
            save(ckpt_path, policy, opt, value, value_opt, episode, update)
    save(ckpt_path, policy, opt, value, value_opt, episode, update)
    envs.close()


def save(path, policy, opt, value, value_opt, episode, update):
    ck = {"policy": policy.state_dict(), "opt": opt.state_dict(), "episode": episode, "update": update,
          "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}
    if value is not None:
        ck["value"] = value.state_dict(); ck["value_opt"] = value_opt.state_dict()
    torch.save(ck, path + ".tmp")
    os.replace(path + ".tmp", path)
    if update % 50 == 0:  # keep periodic snapshots for learning-progress checks
        torch.save(ck, path.replace(".pt", f"_ep{episode}.pt"))


def evaluate(args):
    run_dir = os.path.join(RESULTS, args.run)
    device = torch.device(args.device)
    policy = MLP(args.hidden).to(device)
    ck = torch.load(os.path.join(run_dir, args.ckpt), map_location=device, weights_only=False)
    policy.load_state_dict(ck["policy"])
    np.random.seed(args.seed + 12345)
    envs = make_envs(args.num_envs)
    rewards, first = [], True
    while len(rewards) < args.episodes:
        eps = play_batch(envs, policy, device, seed=args.seed + 12345 if first else None, greedy=args.greedy)
        first = False
        rewards += [float(np.sum(e["r"])) for e in eps]
        print(f"{len(rewards)} episodes, running mean {np.mean(rewards):.2f}", flush=True)
    rewards = np.array(rewards[: args.episodes])
    tag = "greedy" if args.greedy else "stochastic"
    np.save(os.path.join(run_dir, f"eval_{tag}.npy"), rewards)
    print(f"EVAL ({tag}) {len(rewards)} eps: mean {rewards.mean():.3f}  std {rewards.std():.3f}")
    envs.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", choices=["none", "value"], default="none")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value_lr", type=float, default=1e-3)
    p.add_argument("--value_iters", type=int, default=5)
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--max_episodes", type=int, default=100000)
    p.add_argument("--hours", type=float, default=60)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--ckpt", default="ckpt.pt")
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    train(args) if args.mode == "train" else evaluate(args)
