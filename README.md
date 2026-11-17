# Motivation

This GitHub repository is a collection of my Rust implementations of my favorite optimization algorithms.
It contains:
  - a library for linear regression on the CPU with efforts to target the GPU
  - a library for genetic algorithms
  - a library for counterfactual regret minimization
  - a library to optimize Markov processes using counterfactual regret minimization

These crates are not meant for production use, they are meant to enable my research that I find interesting for AI.

# Goal

Gradient-based training of neural networks is the dominant approach in modern AI, and the size and structure of a network is still mostly chosen by hand.
My goal is to build a program that searches for a well-performing neural network architecture for a given dataset at lower training cost than naive search.
My idea is to train a network while periodically masking (switching off) some weights, so that the remaining weights must carry more of the prediction. Choosing which weights to mask at a given point in training is hard, because the effect of a masking decision only becomes visible after further training.

## Decision problem

To make this testable, the idea is framed as a sequential decision problem:

- **Decision point:** at fixed intervals during training, choose a masking action for a group of weights (for example, one layer or one neuron).
- **Actions:** a finite set, e.g. `{keep, mask 10%, mask 25%, mask 50%}` per group, chosen by magnitude or at random.
- **Payoff:** validation loss improvement over the following training window, minus a penalty for the number of active weights.
- **Regret:** for each action, the difference between its payoff and the payoff of the action that was actually taken, accumulated over decision points.

Counterfactual regret minimization (CFR) is designed for two-player zero-sum extensive-form games, where a strategy is learned against an adversary. Training has no adversary, so using CFR here is an assumption that has to be justified by experiment. If the decisions turn out to be effectively independent of each other (no sequential dependence between decision points), a multi-armed bandit method (UCB or Thompson sampling) is the simpler and more appropriate choice. The CFR formulation is kept only if it outperforms that baseline.

## Baselines

Any masking method is compared against these, on the same datasets, architectures, and compute budget:

- **No masking:** standard training to the same number of epochs.
- **Random masking:** masks chosen uniformly at random at the same decision points.
- **Magnitude pruning:** masks the smallest-magnitude weights at the same decision points.
- **Bandit selection:** UCB or Thompson sampling over the same action set.

Related prior work that informs these baselines includes dropout, iterative magnitude pruning and the lottery ticket procedure, and neural architecture search via evolutionary or differentiable methods (the `evol` crate in this repository is a starting point for the evolutionary variant).

## Success criteria

The idea is considered useful if, on at least one benchmark dataset, the CFR-based method reaches the same validation loss as the no-masking baseline with a measurably lower training cost (wall-clock time or total weight updates), and also outperforms the random-masking and bandit baselines. Results should report the mean and spread over several random seeds. If it does not beat these baselines, the project records that negative result and the CFR framing is dropped.

## Scope

This repository is for research. The first milestone is a proof of concept on a small dataset, not a general architecture search tool.

# Current state

- [x] linear regression libary on the CPU
- [x] generic counterfactual regret solver
- [ ] first POC to combine linear regression with counterfactual regret minimization
- [ ] baseline comparison (no masking, random masking, magnitude pruning, bandit selection)

# Outlook

If the success criteria above are met, the approach may reduce neural network training cost. This is an expectation to be tested, not an established result.

