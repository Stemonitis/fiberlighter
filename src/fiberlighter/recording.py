import numpy as np

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
    
    def reset(self):
        self.iso_work = self.iso.copy()
        self.gcamp_work = self.gcamp.copy()
        self.provenance = None