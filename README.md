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
