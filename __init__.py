"""Automatic category suggestion for Wikimedia Commons.

Hybrid retrieval + classification system: an open vision-language encoder,
k-NN label voting over categorised images, zero-shot text prototypes, a
supervised multi-label head trained with asymmetric loss, a metadata scorer,
calibrated stacking and hierarchy-aware post-processing.
"""

__version__ = "1.0.0"
