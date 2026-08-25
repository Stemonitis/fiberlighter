from scipy.signal import butter, filtfilt
from ..registry import register_processing

@register_processing("filtering")
class Filtering:
    def __init__(self, recording):
        self.recording = recording
    def lowpass_filter(self, cutoff=1.0, order=4):
        """Butterworth lowpass filter applied to data. Modifies data in place.

        cutoff : cutoff frequency in Hz above which signal is attenuated. Must be below fs / 2.
        """
        if not 0 < cutoff < self.recording.fs / 2:
            raise ValueError(f"cutoff={cutoff} Hz must be between 0 and {self.recording.fs / 2} Hz for fs={self.recording.fs}")
        b, a = butter(order, cutoff, btype="low", fs=self.recording.fs)
        self.recording.iso_work = filtfilt(b, a, self.recording.iso_work)
        self.recording.gcamp_work = filtfilt(b, a, self.recording.gcamp_work)
