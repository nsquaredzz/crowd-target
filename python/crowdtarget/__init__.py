"""crowdtarget: one target person through a crowd.

Piece 1: identity-switch detection as a conformal test martingale over the tracked
person's appearance scores (martingale.py).  Piece 2: the identity set after
re-emergence with a split-conformal coverage guarantee (identity.py).  Everything
else (SAM 2 stepping, tracklets on public detections, reachable region, exemplar
bank) is plumbing from the literature and is named as such in README.md.
"""
__version__ = "0.0.1"
