# RL-Adventure-2: Policy Gradients

<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg">


PyTorch tutorial of: actor critic / proximal policy optimization / acer / ddpg / twin dueling ddpg / soft actor critic / generative adversarial imitation learning / hindsight experience replay

The deep reinforcement learning community has made several improvements to the [policy gradient](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf) algorithms. This tutorial presents latest extensions in the following order: 

1. Advantage Actor Critic (A2C)
 - [actor-critic.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/1.actor-critic.ipynb)
 - [A3C Paper](https://arxiv.org/pdf/1602.01783.pdf) 
 - [OpenAI blog](https://blog.openai.com/baselines-acktr-a2c/#a2canda3c)
  2. High-Dimensional Continuous Control Using Generalized Advantage Estimation
  - [gae.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/2.gae.ipynb)
  - [GAE Paper](https://arxiv.org/abs/1506.02438)
  3.  Proximal Policy Optimization Algorithms 
  - [ppo.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb)
  - [PPO Paper](https://arxiv.org/abs/1707.06347)
  - [OpenAI blog](https://blog.openai.com/openai-baselines-ppo/)
  4.  Sample Efficient Actor-Critic with Experience Replay 
  - [acer.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/4.acer.ipynb)
  - [ACER Paper](https://arxiv.org/abs/1611.01224)
  5.  Continuous control with deep reinforcement learning
  - [ddpg.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb)
  - [DDPG Paper](https://arxiv.org/abs/1509.02971)
  6. Addressing Function Approximation Error in Actor-Critic Methods
  - [td3.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/6.td3.ipynb)
  - [Twin Dueling DDPG Paper](https://arxiv.org/abs/1802.09477)
  7. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor 
  - [soft actor-critic.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb)
  - [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)
  8.  Generative Adversarial Imitation Learning 
  - [gail.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/8.gail.ipynb)
  - [GAIL Paper](https://arxiv.org/abs/1606.03476)
  9.  Hindsight Experience Replay
  - [her.ipynb](https://github.com/higgsfield/RL-Adventure-2/blob/master/9.her.ipynb)
  - [HER Paper](https://arxiv.org/abs/1707.01495)
  - [OpenAI Blog](https://blog.openai.com/ingredients-for-robotics-research/#understandingher)

# If you get stuck… 
- Remember you are not stuck unless you have spent more than a week on a single algorithm. It is perfectly normal if you do not have all the required knowledge of mathematics and CS.
- Carefully go through the paper. Try to see what is the problem the authors are solving. Understand a high-level idea of the approach, then read the code (skipping the proofs), and after go over the mathematical details and proofs.

# RL Algorithms
Deep Q Learning tutorial: [DQN Adventure: from Zero to State of the Art](https://github.com/higgsfield/RL-Adventure)
[![N|Solid](https://planspace.org/20170830-berkeley_deep_rl_bootcamp/img/annotated.jpg)]()
Awesome RL libs: rlkit [@vitchyr](https://github.com/vitchyr), pytorch-a2c-ppo-acktr [@ikostrikov](https://github.com/ikostrikov),
ACER [@Kaixhin](https://github.com/Kaixhin)

# Best RL courses
- Berkeley deep RL [link](http://rll.berkeley.edu/deeprlcourse/)
- Deep RL Bootcamp [link](https://sites.google.com/view/deep-rl-bootcamp/lectures)
- David Silver's course [link](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- Practical RL [link](https://github.com/yandexdataschool/Practical_RL)

## P3 · Reproducible CPU actor–critic companion (2026-11-09)

The archived actor-critic notebook uses historical `CartPole-v0`/Gym semantics and a PyTorch vector environment. Without changing the original notebook or misrepresenting full A2C reproduction, [`p3/cartpole_actor_critic.py`](p3/cartpole_actor_critic.py) adds a **lightweight, genuine training** companion using modern `CartPole-v1` (Gymnasium 1.1.1) and a NumPy **linear** softmax actor/value critic. It computes discounted returns, advantage gradients, performs actual policy/value parameter updates, and compares greedy evaluation before/after training with a seeded random-action baseline.

```bash
python -m pip install 'gymnasium==1.1.1' 'numpy==1.26.4'
python -m unittest discover -s tests -p 'test_p3_*.py' -v
python -m p3.cartpole_actor_critic --seed 2026 --train-episodes 480 --eval-episodes 64 --output build/p3-cartpole-actor-critic.json
```

CI executes the training **twice**, compares JSON outputs byte-for-byte, checks policy weights changed and that the **held-out** trained policy exceeds the random baseline. A separate 12-episode validation set chooses checkpoints; the 64-episode held-out test seeds are not used for that choice. JSON records versions, seeds, budgets, mean and standard deviation, weight change, and validation trace. This is **linear actor–critic**, not a deep PyTorch A2C/GAE/PPO model, and one fixed seed is not a multi-seed confidence interval. The original notebooks and citations remain intact.

### P3 three-seed replication

The CI also precommits **seeds 2026, 2027, 2028**, each running 480 training episodes and 64 held-out evaluation episodes. [`p3/benchmark.py`](p3/benchmark.py) preserves **each** run and calculates descriptive mean and sample standard deviation across seed-level means, with the same random-action comparison for every seed. The multi-seed JSON is produced twice and compared byte-for-byte. This reduces dependence on one lucky run, but **three seeds do not establish statistical significance or reproduce the original deep A2C notebook**.

```bash
python -m p3.benchmark --output build/p3-multiseed.json
```

### P3 controlled linear-vs-neural comparison

[`p3/torch_actor_critic.py`](p3/torch_actor_critic.py) adds an **actual PyTorch CPU neural actor/critic** with separate one-hidden-layer 32-unit policy and value networks. [`p3/compare_models.py`](p3/compare_models.py) compares it to the NumPy linear baseline using the **same CartPole-v1 reset seeds, 480 training episodes, validation checkpoint frequency, 64 held-out evaluation episodes and random baseline**, for predeclared seeds **2026, 2027, 2028**. Both algorithms use advantage-based Monte Carlo updates; architecture, optimizers and numerical precision differ, so this is a controlled learning benchmark rather than a direct framework-only or original notebook A2C comparison.

```bash
python -m pip install "gymnasium==1.1.1" "numpy==1.26.4"
python -m pip install "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -p "neural_contract_test.py" -v
python -m p3.compare_models --output build/p3-model-comparison.json --timing-output build/p3-model-timing.json
```

The new CI checks nonzero actor **and** critic gradients, fixed-seed repeatability and full comparison JSON byte-for-byte stability. It **does not require the neural network to outperform** the linear baseline. Runtime is recorded separately because timing is not byte-reproducible and includes training, validation and evaluation, but excludes dependency installation. It is single-machine sequential timing, **not** a hardware-standardized benchmark. Neither implementation reproduces the original vectorized n-step PyTorch A2C notebook; that remains an open research task.

### Verified comparison result (2026-11-09)

The [CPU comparison workflow](https://github.com/1998x-stack/RL-Adventure-2/actions/runs/42637469330) succeeded after separately fixing (1) PyTorch test isolation from the dependency-light NumPy job and (2) float64 normalization when passing softmax probabilities to NumPy's action sampler. **Both failures remain visible in the workflow history.**

| Seed | NumPy linear: mean reward | PyTorch neural: mean reward | Neural minus linear |
|---|---:|---:|---:|
| 2026 | 382.109 | 324.297 | -57.812 |
| 2027 | 410.953 | 280.266 | -130.687 |
| 2028 | 484.156 | 436.828 | -47.328 |
| Mean over three seed-level means | **425.739** | **347.130** | -78.609 |
| Sample standard deviation | 52.606 | 80.740 | |

- **Training/evaluation:** same CartPole-v1, seeds 2026/2027/2028, 480 training episodes per seed, 12 validation episodes per checkpoint, 64 held-out evaluation episodes per seed.
- **Model size:** 15 trainable values in the NumPy linear actor/critic versus 483 neural network parameters (two independent one-hidden-layer networks with 32 Tanh units).
- **Measured single-host sequential runtimes:** NumPy 3.667/3.875/6.063 seconds; PyTorch 11.648/9.437/15.554 seconds. The per-seed averages are approximately 4.54 s vs 12.21 s. These observations include training, validation and evaluation, exclude dependency install, and **are not hardware-normalized**.
- **Reproducibility:** three neural gradient/seed tests passed; deterministic comparison JSON matched byte-for-byte on two complete executions. Wall-clock numbers deliberately live in a separate non-deterministic JSON file.
- **Important limitation:** training algorithm details, architecture, optimizer and arithmetic precision differ. The results **do not establish causality**, general superiority of linear networks, or parity with the original vectorized n-step A2C Notebook. Only the two modern companion implementations were compared, and performance is based on three seeds.


### P3: notebook-style vectorized n-step A2C (modern Gymnasium)

The archived \`1.actor-critic.ipynb\` uses **16 environments**, **5-step bootstrapped returns**, separate **256-unit ReLU** policy/value networks, advantage-weighted actor loss, value loss and entropy regularization. [\`p3/vector_nstep_a2c.py\`](p3/vector_nstep_a2c.py) now preserves that algorithmic structure on modern \`CartPole-v1\` and PyTorch CPU. The implementation uses \`SyncVectorEnv\` rather than the original subprocess helper, and \`AutoresetMode.SAME_STEP\` to ensure every training step is a real transition. It bootstraps from \`final_obs\` on Gymnasium time-limit truncations instead of accidentally using the reset observation.

The workflow runs three fixed seeds (2026/2027/2028), **16 × 5 × 4,000 = 320,000 environment transitions per seed (20,000 vector steps as in the original notebook)**, validates checkpoints on a separate 12-episode seed set and evaluates on 64 held-out episodes per seed. It tests actual policy/value updates, SAME_STEP semantics and n-step masks; it byte-compares two result JSON exports. Wall times are kept in a separate file because they cannot be exactly reproduced.

\`\`\`bash
python -m pip install "gymnasium==1.1.1" "numpy==1.26.4"
python -m pip install "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -p "vector_a2c_contract_test.py" -v
python -m p3.vector_nstep_a2c --updates 4000 --eval-episodes 64
\`\`\`

**Scope:** this is an audited, modern **notebook-style** vectorized A2C, **not** a verbatim rerun of the 2018 \`CartPole-v0\`/16-worker \`SubprocVecEnv\` experiment. It uses synchronous stepping, 3e-4 Adam, separate Gymnasium time-limit bootstrapping and clipped gradient norms. Its training budget is *environment transitions*, whereas earlier linear/neural comparisons budgeted *480 completed episodes*, so those published reward numbers must not be interpreted as equal-sample head-to-head rankings. More seeds, a matched environment-frame budget, and an actual subprocess vector test would be necessary for rigorous algorithm comparison.
