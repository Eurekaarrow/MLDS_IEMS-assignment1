# MLDS Assignment 1

Code for the shuttle dispatch DP problem and REINFORCE on CartPole / Pong.

## Setup
```
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

## Part 2: value / policy iteration
```
python part2_dp/shuttle.py       # enumeration, VI, PI + plots in results/dp/
python part2_dp/validate_dp.py   # validation tests (prints PASS/FAIL)
```

## Part 3: policy gradient
CartPole (runs on CPU, ~1-2 min per seed):
```
python part3_pg/cartpole.py train --baseline none  --seeds 0 1 2 3 4 --episodes 2000 --lr 1e-3 --scale --tag _C_lr1e-3_scale
python part3_pg/cartpole.py train --baseline value --seeds 0 1 2 3 4 --episodes 2000 --lr 1e-3 --scale --tag _C_lr1e-3_scale
python part3_pg/cartpole.py eval  --baseline value --seeds 0 --tag _C_lr1e-3_scale
python part3_pg/validate_cartpole.py
```
`--baseline mean` is the constant baseline, `--baseline value` is the learned V(s) baseline.

Pong (one GPU, ~8 h for 30k episodes):
```
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
python part3_pg/pong.py train --baseline none  --run pong_nobase_lr1e-2 --lr 1e-2 --max_episodes 30000
python part3_pg/pong.py train --baseline value --run pong_value_lr1e-2  --lr 1e-2 --max_episodes 30000
python part3_pg/pong.py eval  --run pong_value_lr1e-2 --episodes 500
python part3_pg/validate_pong.py
```

Plots: `python part3_pg/plots.py cartpole|pong|sweep` (see the top of plots.py).
Checkpoints and training logs are not in the repo (too big).

Figures used in the report are in `results/figures/`, `results/dp/` and the `validation` folders.
