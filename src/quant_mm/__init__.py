"""quant_mm: a limit-order-book matching engine and RL market maker.

Subpackages
-----------
lob     : the matching engine (price-time priority, partial fills, cancels).
sim     : market simulators (synthetic stochastic flow + historical replay).
agents  : quoting policies (Avellaneda-Stoikov baseline, DQN, PPO).
eval    : performance metrics and plotting.
"""

__version__ = "0.1.0"
