import numpy as np
from typing import Literal

from .registry import PROCESSORS

from .preprocessing.bleach_correction import BleachCorrection
from .preprocessing.motion_correction import MotionCorrection
from .preprocessing.noise_correction import NoiseCorrection
from .preprocessing.normalization import Normalization
from .preprocessing.diagnostics import Diagnostics
from .visualization.plots import Visualization

#add how long is the recordings
#add recording frequency in Hz
#add if the units are raw data of normalized and if normalized then how
#add provenance to enable plotting of the fitted baselines and such 
#add the spectral profile so that the bleach correction and other methods can use it for optimizing
class Recording:
    def __init__(
        self,
        iso: np.ndarray,
        gcamp: np.ndarray,
        time: np.ndarray,
        events: dict[str, np.ndarray] | None = None,
        provenance=None,
        fs = None,
        unit_type: Literal["raw", "deltaF", "df/f", "df/f_percent", "z_score"] = "raw",
        baseline_gcamp: np.ndarray | None = None,
        baseline_iso: np.ndarray | None = None
    ):
        self.iso = iso
        self.gcamp = gcamp
        self.time = time
        self.events = events or {}
        self.provenance = provenance
        self.unit_type = unit_type
        self.bleach_fits = {}
        self.baseline_gcamp = baseline_gcamp
        self.baseline_iso = baseline_iso

        self.fs = 1 / np.median(np.diff(time)) #if fs is none()

        self.iso_work = iso.copy()
        self.gcamp_work = gcamp.copy()

        for name, processor_class in PROCESSORS.items():
            setattr(self, name, processor_class(self))
    
    @property
    def unit_type(self):
        return self._unit_type

    @unit_type.setter
    def unit_type(self, value):
        # Existing whole-recording processors set both channels together.
        self._unit_type = value
        self.channel_units = {"iso": value, "gcamp": value}

    def _sync_units(self):
        units = set(self.channel_units.values())
        self._unit_type = next(iter(units)) if len(units) == 1 else "mixed"

    def reset(self):
        self.iso_work = self.iso.copy()
        self.gcamp_work = self.gcamp.copy()
        self.unit_type = "raw"
        self.provenance = None
        self.baseline_iso = None
        self.baseline_gcamp = None
        self.bleach_fits.clear()
        return self