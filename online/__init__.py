"""
Online fine-tuning of the Skat Decision Transformer via self-play.

Follows the Online Decision Transformer (Zheng et al., 2022, https://arxiv.org/abs/2202.05607):
a DT pre-trained offline is fine-tuned on trajectories it collects itself, using a stochastic, entropy-regularised
policy, a trajectory-level replay buffer and hindsight return relabelling.
"""
