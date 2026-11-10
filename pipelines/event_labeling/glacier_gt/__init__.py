"""Glacier event ground truth: measurement layer + window labels + evidence figures + review."""

CLASSES = ["Stable", "Retreat", "Advance", "Fragmentation", "Disappearance"]
NON_CLASSES = ["Ambiguous", "Insufficient"]
PRECEDENCE = ["Disappearance", "Fragmentation", "Retreat", "Advance", "Stable"]
