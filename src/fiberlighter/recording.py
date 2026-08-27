import numpy as np
from .registry import PROCESSORS

from .preprocessing.bleach_correction import BleachCorrection
from .preprocessing.motion_correction import MotionCorrection
from .preprocessing.noise_correction import NoiseCorrection
from .visualization.plots import Visualization


class Recording:
    def __init__(
        self,
        iso: np.ndarray,
        gcamp: np.ndarray,
        time: np.ndarray,
        events: dict[str, np.ndarray] | None = None,
        provenance=None,
        fs = None
       
    ):
        self.iso = iso
        self.gcamp = gcamp
        self.time = time
        self.events = events or {}
        self.provenance = provenance

        self.fs = 1 / np.median(np.diff(time)) #if fs is none()

        self.iso_work = iso.copy()
        self.gcamp_work = gcamp.copy()

        for name, processor_class in PROCESSORS.items():
            setattr(self, name, processor_class(self))
    
    def reset(self):
        self.iso_work = self.iso.copy()
        self.gcamp_work = self.gcamp.copy()
        self.provenance = None