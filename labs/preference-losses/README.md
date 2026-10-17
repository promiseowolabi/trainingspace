# Lab 13 — Preference and RL Losses

Lessons: F1.L4 (reward models), F1.L5 (RLHF and DPO), F1.L6 (GRPO).

You edit one file: `prefs.py` — eight short functions, from summed sequence
log-probabilities to DPO, PPO's clipped objective, GRPO advantages and the k3
KL estimator. `log_sigmoid` is given.

```sh
pytest
PREFIX=<you> python verify.py
```

Checks: sequence-logprob · bradley-terry · dpo-loss · dpo-gradient ·
ppo-clip · grpo-advantages · kl-k3
