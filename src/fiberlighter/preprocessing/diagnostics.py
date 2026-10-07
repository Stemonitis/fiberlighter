"""Diagnostics.

does not return the recording object, but instead returns a dictionary of diagnostic information or a plot and stuff
TODO / known issues
-------------------
"""
import numpy as np
from scipy.signal import periodogram, spectrogram, coherence
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ..registry import register_processing


@register_processing("diagnostics")
class Diagnostics:
    def __init__(self, recording):
        self.recording = recording

    def _plot_raw_traces(self, ax):
        """Add the two raw channels to a shared time axis in minutes."""
        rec = self.recording
        minutes = (np.asarray(rec.time) - rec.time[0]) / 60
        ax.plot(minutes, rec.gcamp, color="#7556c7", lw=0.8, label="GCaMP")
        ax.plot(minutes, rec.iso, color="#168c87", lw=0.8, label="Iso")
        ax.set_ylabel("Recorded intensity\n(source units)")
        ax.set_title("Full recording", loc="left")
        ax.legend(frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    # def summary
#     Recording duration, sample count, estimated sampling rate, and timestamp gaps.
# Missing/nonfinite samples and constant channels.
# Mean, standard deviation, minimum, and maximum for each channel.

    def frequency_analysis_whole(self, heatmap=True):
        """Plot full raw traces and their full-record FFT-based spectra.

        Spectra are shown in dB relative to each channel's own largest PSD bin:
        0 dB is its peak; -10 dB is one tenth of that power density. This compares
        spectral shape, not absolute channel power. Values below -100 dB are
        clipped for display. No recording arrays are changed.
        With heatmap=True, show the same spectrum as one colored row per
        channel, using a fixed -100 to 0 dB scale and a separate peak per row.

        Shading divides illustrative timescales (>10 min, 10 s–10 min, <10 s).
        These are not validated bleaching/GCaMP boundaries or filter cutoffs.
        Bleaching, biological changes, and artifacts can overlap in frequency.
        """
        rec = self.recording
        time = np.asarray(rec.time, dtype=float)
        channels = (("GCaMP", rec.gcamp, "#7556c7"), ("Iso", rec.iso, "#168c87"))
        if time.ndim != 1 or time.size < 4 or not np.all(np.isfinite(time)):
            raise ValueError("time must contain at least four finite samples")
        if not np.isfinite(rec.fs) or rec.fs <= 0 or not np.allclose(
            np.diff(time), 1 / rec.fs, rtol=0.01, atol=0
        ):
            raise ValueError("Frequency analysis needs evenly spaced timestamps consistent with fs")
        for name, signal, _ in channels:
            if np.shape(signal) != time.shape or not np.all(np.isfinite(signal)):
                raise ValueError(f"{name} must be finite and match the timestamps")

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.8]}
        )
        fig.patch.set_facecolor("#faf9f6")
        for ax in axes:
            ax.set_facecolor("#ffffff")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#d0d4d9")
            ax.tick_params(colors="#485361", labelsize=9)
            ax.grid(axis="y", color="#e8ebef", linewidth=0.7)
            ax.set_axisbelow(True)

        spectra = []
        for name, signal, color in channels:
            axes[0].plot((time - time[0]) / 60, signal, color=color, lw=0.8, label=name)
            frequency, power = periodogram(
                signal, fs=rec.fs, window="hann", detrend="constant", scaling="density"
            )
            frequency, power = frequency[1:], power[1:]
            spectra.append(10 * np.log10(np.maximum(power / power.max(), 1e-10))
                           if power.max() > 0 else np.full_like(power, np.nan))
            if power.max() > 0:
                relative_db = 10 * np.log10(np.maximum(power / power.max(), 1e-10))
                axes[1].plot(frequency, relative_db, color=color, lw=0.8, alpha=0.8, label=name)
            else:
                axes[1].plot([], [], color=color, label=f"{name}: no fluctuation power")

        if heatmap:
            axes[1].clear()
            df = rec.fs / time.size
            edges = np.r_[frequency - df / 2, rec.fs / 2]
            cmap = plt.get_cmap("turbo").copy()
            cmap.set_bad("#d9d9d9")
            mesh = axes[1].pcolormesh(
                edges, [0, 1, 2], np.ma.masked_invalid(spectra),
                cmap=cmap, vmin=-100, vmax=0, rasterized=True,
            )
            axes[1].set(xscale="log", xlim=(frequency[0], rec.fs / 2),
                        ylim=(2, 0), xlabel="Frequency · corresponding period")
            axes[1].set_yticks([0.5, 1.5], ["GCaMP", "Iso"])
            ticks = [(1 / 3600, "1 hour"), (1 / 600, "10 min"),
                     (1 / 60, "1 min"), (.1, "10 sec"), (1, "1 sec"), (10, "0.1 sec")]
            ticks = [(f, label) for f, label in ticks if frequency[0] <= f <= rec.fs / 2]
            if ticks:
                axes[1].set_xticks([f for f, _ in ticks],
                                  [f"{f:.2g} Hz\n{label}" for f, label in ticks])
            axes[1].set_title("Full-record frequency power · blue = low, red = high")
            fig.colorbar(mesh, ax=axes[1], label="Power relative to each channel's peak (dB)")
            axes[0].set(xlabel="Time (minutes)", ylabel="Recorded intensity\n(source units)")
            axes[0].legend(frameon=False)
            fig.suptitle("Frequency overview · raw channels")
            fig.text(.08, .02, "0 dB = channel peak; −10 dB = one tenth of peak power density. "
                     "Gray = undefined (constant channel).", fontsize=9)
            fig.tight_layout(rect=(0, .07, 1, .95))
            return fig, axes

        bands = (
            (frequency[0], 1 / 600, "Very slow · >10 min\nBleaching / drift possible\nSlow biology can overlap", "#f4dfb5"),
            (1 / 600, 0.1, "10 seconds–10 minutes\nGCaMP changes possible\nDrift / artifacts can overlap", "#dceee4"),
            (0.1, rec.fs / 2, "Faster than 10 seconds\nGCaMP transients possible\nArtifacts / noise can overlap", "#e1e8f5"),
        )
        for left, right, label, color in bands:
            left, right = max(left, frequency[0]), min(right, rec.fs / 2)
            if right > left:
                axes[1].axvspan(left, right, color=color, alpha=0.65, zorder=0)
                if right / left > 2:
                    axes[1].text(
                        np.sqrt(left * right), 0.97, label,
                        transform=axes[1].get_xaxis_transform(), ha="center", va="top",
                        fontsize=8.5, color="#475364", linespacing=1.5,
                    )

        ticks = [(1 / 3600, "1 hour"), (1 / 600, "10 min"), (1 / 60, "1 min"),
                 (0.1, "10 sec"), (1, "1 sec"), (10, "0.1 sec"), (100, "0.01 sec")]
        ticks = [(f, label) for f, label in ticks if frequency[0] <= f <= rec.fs / 2]
        axes[1].set_xscale("log")
        axes[1].set_xlim(frequency[0], rec.fs / 2)
        axes[1].set_ylim(-100, 30)
        if ticks:
            axes[1].set_xticks([f for f, _ in ticks])
            axes[1].set_xticklabels([f"{f:.4g} Hz\n{label}" for f, label in ticks])
        axes[1].set_yticks([-100, -80, -60, -40, -20, 0])
        axes[0].set_title("Full recording", loc="left", fontweight="bold", color="#273444")
        axes[0].set_xlabel("Time (minutes)")
        axes[0].set_ylabel("Recorded intensity\n(source units)")
        axes[0].legend(loc="upper right", frameon=False, ncol=2)
        axes[1].set_title("Full-record frequency spectrum", loc="left", fontweight="bold", color="#273444")
        axes[1].set_xlabel("Frequency · corresponding period", labelpad=10)
        axes[1].set_ylabel("Power relative to channel peak (dB)")
        axes[1].legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
        fig.suptitle(
            f"Frequency overview   |   {(time[-1] - time[0]) / 60:.1f} min   |   {rec.fs:.3g} samples/s",
            fontsize=15, fontweight="bold", color="#273444", x=0.08, ha="left",
        )
        fig.text(
            0.08, 0.025,
            "Shaded bands are timescale guides, not signal identification or filter recommendations.\n"
            "Each spectrum has its own 0 dB reference; absolute power between channels is not compared.",
            fontsize=9, color="#627080", linespacing=1.5,
        )
        fig.tight_layout(rect=(0, 0.095, 1, 0.95), h_pad=2)
        return fig, axes

    def frequency_analysis_timeframed(self, window_seconds=600):
        """Show raw-channel power over time using overlapping Hann windows.

        Each window has its mean removed. Colors express power density as a
        percentage of the strongest cell in that channel's entire heatmap,
        not a percentage of total power. The logarithmic color scale shows
        weak components too; values below 0.000001% share the darkest color.
        Shorter windows improve time localization but resolve fewer slow
        frequencies. Windows longer than the recording are shortened to fit.
        Only complete windows are used; blank edges have no window centers.
        Returns (fig, axes): raw traces, GCaMP heatmap, then Iso heatmap.
        """
        rec = self.recording
        time = np.asarray(rec.time, dtype=float)
        if time.ndim != 1 or time.size < 4 or not np.all(np.isfinite(time)):
            raise ValueError("time must contain at least four finite samples")
        if not np.isfinite(rec.fs) or rec.fs <= 0 or not np.allclose(
            np.diff(time), 1 / rec.fs, rtol=0.01, atol=0
        ):
            raise ValueError("Frequency analysis needs evenly spaced timestamps consistent with fs")
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        nperseg = min(time.size, round(window_seconds * rec.fs))
        if nperseg < 4:
            raise ValueError("The analysis window must contain at least four samples")
        channels = (("GCaMP", rec.gcamp), ("Iso", rec.iso))
        for name, signal in channels:
            if np.shape(signal) != time.shape or not np.all(np.isfinite(signal)):
                raise ValueError(f"{name} must be finite and match the timestamps")

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.5, 1.5]},
                                 layout="constrained")
        fig.patch.set_facecolor("#faf9f6")
        self._plot_raw_traces(axes[0])
        for ax, (name, signal) in zip(axes[1:], channels):
            frequency, centers, power = spectrogram(
                signal, fs=rec.fs, window="hann", nperseg=nperseg,
                noverlap=nperseg // 2, detrend="constant", scaling="density",
            )
            frequency, power = frequency[1:], power[1:]
            peak = power.max()
            relative = 100 * power / peak if peak > 0 else np.zeros_like(power)
            # Explicit cell edges also support a recording with just one window.
            half_step = (nperseg - nperseg // 2) / rec.fs / 2
            time_edges = np.r_[centers - half_step, centers[-1] + half_step] / 60
            df = rec.fs / nperseg
            frequency_edges = np.r_[frequency - df / 2, rec.fs / 2]
            mesh = ax.pcolormesh(
                time_edges, frequency_edges, np.maximum(relative, 1e-6),
                cmap="turbo", norm=LogNorm(vmin=1e-6, vmax=100), rasterized=True,
            )
            ax.set_yscale("log")
            ticks = [(1 / 600, "10 min"), (1 / 60, "1 min"),
                     (0.1, "10 sec"), (1, "1 sec"), (10, "0.1 sec")]
            ticks = [(f, label) for f, label in ticks if frequency_edges[0] <= f <= rec.fs / 2]
            ax.set_yticks([f for f, _ in ticks],
                          [f"{label} ({f:.2g} Hz)" for f, label in ticks])
            ax.minorticks_off()
            ax.set_ylabel("Period · frequency")
            ax.set_title(name if peak > 0 else f"{name} — no fluctuation power", loc="left")
            ax.set_xlim(0, (time[-1] - time[0]) / 60)
        colorbar = fig.colorbar(mesh, ax=axes, pad=0.02)
        colorbar.set_ticks([1e-6, 1e-4, .01, 1, 100],
                           labels=["≤0.000001%", "0.0001%", "0.01%", "1%", "100%"])
        colorbar.set_label("Power relative to each channel's peak (%) · log scale")
        axes[-1].set_xlabel("Time (minutes)")
        fig.suptitle(
            f"Frequency heatmap · {nperseg / rec.fs / 60:.2g}-minute windows, 50% overlap\n"
            "Blue = low power, red = high power; channels have separate references",
            fontsize=13,
        )
        return fig, axes

    def coherence_whole(self, window_seconds=120, heatmap=True):
        """Return (fig, ax) for coherence averaged across the full recording."""
        return self.coherence(window_seconds=window_seconds, heatmap=heatmap)

    def coherence_timeframed(self, window_seconds=120, time_window_seconds=600):
        """Return (fig, axes): raw traces above a coherence heatmap over time.

        window_seconds sets the FFT segment length; time_window_seconds sets
        each local region in which multiple segments are averaged.
        """
        return self.coherence(window_seconds=window_seconds,
                              time_window_seconds=time_window_seconds)

    def coherence(self, window_seconds=600, heatmap=False, time_window_seconds=None):
        """Plot magnitude-squared coherence between the two raw channels.

        Uses Welch averaging with Hann windows, 50% overlap, and per-window
        mean removal. Returns (fig, ax), without changing recording data.
        Set heatmap=True for a single-row frequency heatmap of the same
        estimates. This summarizes the recording, not changes over time.
        Set time_window_seconds to compute a time–frequency heatmap instead.
        That mode returns (fig, axes), with raw traces above the heatmap.
        Each outer time window averages several shorter window_seconds
        segments. Both levels use 50% overlap. Only complete outer windows
        are used; edge regions without estimates are blank. Locally constant
        channels or undefined frequency bins are gray.
        Values near 1 indicate a consistent linear relationship at that
        frequency, not equal amplitudes or proof of an artifact. Values near
        0 indicate a weak estimated relationship. No significance threshold
        is implied. Longer windows resolve slower frequencies but leave fewer
        segments to average, making estimates less reliable.

        Require at least three overlapping windows: a single-window estimate
        is trivially 1 wherever both signals have power. Three is a minimum
        guard, not a guarantee of a reliable estimate.
        """
        rec = self.recording
        time = np.asarray(rec.time, dtype=float)
        if time.ndim != 1 or time.size < 8 or not np.all(np.isfinite(time)):
            raise ValueError("time must contain at least eight finite samples")
        if not np.isfinite(rec.fs) or rec.fs <= 0 or not np.allclose(
            np.diff(time), 1 / rec.fs, rtol=0.01, atol=0
        ):
            raise ValueError("Coherence needs evenly spaced timestamps consistent with fs")
        for name, signal in (("GCaMP", rec.gcamp), ("Iso", rec.iso)):
            if np.shape(signal) != time.shape or not np.all(np.isfinite(signal)):
                raise ValueError(f"{name} must be finite and match the timestamps")
            if np.ptp(signal) == 0:
                raise ValueError(f"Coherence is undefined for a constant {name} channel")
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be finite and positive")
        nperseg = round(window_seconds * rec.fs)
        if nperseg < 4 or nperseg > time.size // 2:
            raise ValueError(
                "Choose a window containing at least four samples and no more "
                "than half the recording, so multiple windows can be averaged"
            )
        overlap = nperseg // 2
        if time_window_seconds is not None:
            if not np.isfinite(time_window_seconds) or time_window_seconds <= 0:
                raise ValueError("time_window_seconds must be finite and positive")
            outer = round(time_window_seconds * rec.fs)
            if outer < 2 * nperseg or outer > time.size:
                raise ValueError("time_window_seconds must fit the recording and be at least "
                                 "twice window_seconds to average multiple segments")
            hop = outer - outer // 2
            starts = np.arange(0, time.size - outer + 1, hop)
            frequency = np.fft.rfftfreq(nperseg, d=1 / rec.fs)[1:]
            values = np.full((frequency.size, starts.size), np.nan)
            for column, start in enumerate(starts):
                gcamp = np.asarray(rec.gcamp)[start:start + outer]
                iso = np.asarray(rec.iso)[start:start + outer]
                if np.ptp(gcamp) == 0 or np.ptp(iso) == 0:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    _, estimate = coherence(
                        gcamp, iso, fs=rec.fs, window="hann", nperseg=nperseg,
                        noverlap=overlap, detrend="constant",
                    )
                values[:, column] = estimate[1:]
            values = np.ma.masked_invalid(np.where(np.isfinite(values),
                                                   np.clip(values, 0, 1), np.nan))
            centers = (starts + outer / 2) / rec.fs
            time_edges = np.r_[centers - hop / rec.fs / 2,
                               centers[-1] + hop / rec.fs / 2] / 60
            edges = np.r_[frequency - rec.fs / nperseg / 2, rec.fs / 2]
            cmap = plt.get_cmap("turbo").copy()
            cmap.set_bad("#d9d9d9")
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 2]},
                                     layout="constrained")
            self._plot_raw_traces(axes[0])
            ax = axes[1]
            fig.patch.set_facecolor("#faf9f6")
            mesh = ax.pcolormesh(time_edges, edges, values, cmap=cmap,
                                 vmin=0, vmax=1, rasterized=True)
            ax.set(yscale="log", xlabel="Time (minutes)", ylabel="Period · frequency",
                   xlim=(0, (time[-1] - time[0]) / 60))
            ticks = [(1 / 3600, "1 hour"), (1 / 600, "10 min"),
                     (1 / 60, "1 min"), (.1, "10 sec"), (1, "1 sec"), (10, "0.1 sec")]
            ticks = [(f, label) for f, label in ticks if frequency[0] <= f <= rec.fs / 2]
            if ticks:
                ax.set_yticks([f for f, _ in ticks],
                             [f"{label} ({f:.2g} Hz)" for f, label in ticks])
            ax.minorticks_off()
            segments = 1 + (outer - nperseg) // (nperseg - overlap)
            ax.set_title(
                f"GCaMP–Iso coherence over time · {outer / rec.fs / 60:.2g}-minute regions\n"
                f"{segments} segments of {nperseg / rec.fs:.3g} s per estimate · "
                "gray = undefined; shared structure does not identify artifacts"
            )
            fig.colorbar(mesh, ax=axes, label="Coherence · 0 = weak, 1 = strong")
            return fig, axes

        segments = 1 + (time.size - nperseg) // (nperseg - overlap)
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency, shared = coherence(
                rec.gcamp, rec.iso, fs=rec.fs, window="hann",
                nperseg=nperseg, noverlap=overlap, detrend="constant",
            )
        frequency, shared = frequency[1:], shared[1:]
        # Frequencies with undefined coherence remain gaps, not zeroes.
        shared = np.where(np.isfinite(shared), np.clip(shared, 0, 1), np.nan)
        fig, ax = plt.subplots(figsize=(11, 3 if heatmap else 4.5), layout="constrained")
        fig.patch.set_facecolor("#faf9f6")
        if heatmap:
            df = rec.fs / nperseg
            edges = np.r_[frequency - df / 2, rec.fs / 2]
            cmap = plt.get_cmap("turbo").copy()
            cmap.set_bad("#d9d9d9")
            mesh = ax.pcolormesh(
                edges, [0, 1], np.ma.masked_invalid(shared[None, :]),
                cmap=cmap, vmin=0, vmax=1, rasterized=True,
            )
            ax.set_xscale("log")
            ax.set_yticks([0.5], ["GCaMP–Iso"])
            fig.colorbar(mesh, ax=ax, label="Coherence · 0 = weak, 1 = strong")
        else:
            ax.semilogx(frequency, shared, color="#7556c7", lw=1)
            ax.set(ylim=(0, 1.05), ylabel="Magnitude-squared coherence (0–1)")
            ax.grid(axis="y", alpha=0.2)
        ax.set(xlim=(frequency[0], frequency[-1]),
               xlabel="Frequency · corresponding period")
        ticks = [(1 / 3600, "1 hour"), (1 / 600, "10 min"),
                 (1 / 60, "1 min"), (0.1, "10 sec"), (1, "1 sec"), (10, "0.1 sec")]
        ticks = [(f, label) for f, label in ticks if frequency[0] <= f <= frequency[-1]]
        if ticks:
            ax.set_xticks([f for f, _ in ticks],
                         [f"{f:.2g} Hz\n{label}" for f, label in ticks])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(
            f"GCaMP–Iso coherence · {nperseg / rec.fs:.3g} s windows · {segments} segments\n"
            "Shared frequency structure does not identify artifacts",
            loc="left",
        )
        return fig, ax

    def byb(self, channel="gcamp", period_seconds=(10, 300),
                             time_bin_seconds=300, **kwargs):
        """Plot cycle counts, BOSC-style occupancy and unweighted EMD occupancy.

        Returns (fig, axes, results). See oscillation_comparison.py for method
        definitions, thresholds and limitations. Raw recording is not modified.
        """
        from .oscillation_comparison import compare_oscillations
        return compare_oscillations(self.recording, channel, period_seconds,
                                    time_bin_seconds, **kwargs)

    # Keep the original public name used by the individual method wrappers.
    compare_oscillations = byb

    def bycycle_analysis(self, channel="gcamp", period_seconds=(10, 300),
                         time_bin_seconds=300, **kwargs):
        """Plot accepted cycle counts, whole-recording and timeframed.

        Returns (fig, axes, results); uses the selected raw channel.
        """
        return self.compare_oscillations(channel, period_seconds, time_bin_seconds,
                                         method="bycycle", **kwargs)

    def bosc_analysis(self, channel="gcamp", period_seconds=(10, 300),
                      time_bin_seconds=300, **kwargs):
        """Plot basic BOSC-style occupancy (not eBOSC/fBOSC).

        Returns (fig, axes, results); uses the selected raw channel.
        """
        return self.compare_oscillations(channel, period_seconds, time_bin_seconds,
                                         method="bosc", **kwargs)

    def emd_analysis(self, channel="gcamp", period_seconds=(10, 300),
                     time_bin_seconds=300, **kwargs):
        """Plot unweighted EMD frequency occupancy in component-seconds.

        Returns (fig, axes, results); uses the selected raw channel.
        """
        return self.compare_oscillations(channel, period_seconds, time_bin_seconds,
                                         method="emd", **kwargs)
