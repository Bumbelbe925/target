"""Detector de cognitive targets: círculos de 5 anillos de color.

    from cognitive import CognitiveDetector
    det = CognitiveDetector()
    for t in det.detect(imagen_bgr):
        print(t.label, t.total, t.colors)
"""

from .config import Config
from .pipeline import CognitiveDetector, TargetDetection
from .scoring import FAKE, HARMED, STABLE, UNHARMED, VALUES, label_for

__all__ = ["CognitiveDetector", "TargetDetection", "Config", "label_for", "VALUES",
           "HARMED", "STABLE", "UNHARMED", "FAKE"]
