# EEG quality checker

import numpy as np
import mne
from scipy.signal import welch, butter, sosfiltfilt, find_peaks

# electrodes near the eyes, blinks show up big here
FRONT_CHANNELS = ["Fp1", "Fp2", "Fpz", "AF3", "AF4", "AF7", "AF8", "AFz"]


# loading data

def load_recording(path):
    # for your own .edf file
    # returns data (channels x samples), sampling rate, channel names
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    data = raw.get_data() * 1e6              # volts -> microvolts
    return data, raw.info["sfreq"], raw.ch_names


def load_eegbci(subject=1, run=1):
    # sample data from physionet, run 1 = eyes open, run 2 = eyes closed
    from mne.datasets import eegbci
    path = eegbci.load_data(subject, [run])[0]
    raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
    eegbci.standardize(raw)                  # fixes names like "Fc5." -> "FC5"
    data = raw.get_data() * 1e6
    return data, raw.info["sfreq"], raw.ch_names


def make_fake_eeg(duration=30, sfreq=250, flat_ch=None, noisy_ch=None,
                  line_hz=None, blink_times=(), seed=0):
    # fake 8 channel eeg just for testing my checks
    # I can break it on purpose:
    #   flat_ch / noisy_ch = which channel number to ruin
    #   line_hz = add 50 or 60 Hz hum
    #   blink_times = seconds where blinks happen on Fp1 and Fp2
    rng = np.random.default_rng(seed)
    ch_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    t = np.arange(int(duration * sfreq)) / sfreq    # time of each sample

    data = rng.normal(0, 10, (len(ch_names), len(t)))   # random noise as base
    data += 8 * np.sin(2 * np.pi * 10 * t)              # 10 Hz wave to look like alpha

    if noisy_ch is not None:
        data[noisy_ch] += rng.normal(0, 150, len(t))
    if line_hz is not None:
        data += 15 * np.sin(2 * np.pi * line_hz * t)    # same hum on every channel
    for bt in blink_times:
        bump = 150 * np.exp(-((t - bt) ** 2) / (2 * 0.1 ** 2))   # smooth bump = blink
        data[0] += bump
        data[1] += bump
    if flat_ch is not None:
        data[flat_ch] = 0.0          # do this last so nothing gets added on top

    return data, sfreq, ch_names


# check 1: flat or noisy channels

def check_flat_noisy(data, sfreq, ch_names, flat_thresh=0.5, noisy_thresh=100, win_sec=1.0):
    # split the recording into 1 second windows
    # measure how much each channel moves in each window (std in microvolts)
    # then take the median, so a couple of blinks don't make a good channel look bad
    win = int(win_sec * sfreq)
    n_win = data.shape[1] // win
    chunks = data[:, : n_win * win].reshape(data.shape[0], n_win, win)
    chunks = chunks - chunks.mean(axis=2, keepdims=True)   # remove the offset

    win_std = chunks.std(axis=2)                       # channels x windows
    typical = np.median(win_std, axis=1)               # normal level of each channel
    bad_frac = (win_std > noisy_thresh).mean(axis=1)   # how many windows were huge

    results = []
    for name, s, b in zip(ch_names, typical, bad_frac):
        if s < flat_thresh:
            status = "flat"      # barely moves, probably disconnected
        elif s > noisy_thresh:
            status = "noisy"     # moves way too much
        else:
            status = "ok"
        results.append({
            "channel": name,
            "typical_std_uv": round(float(s), 2),
            "bad_windows_pct": round(float(b) * 100, 1),
            "status": status,
        })
    return results


# check 2: 50/60 Hz power line noise

def check_line_noise(data, sfreq, ch_names, ratio_thresh=3.0):
    # mains electricity shows up as a sharp spike at 50 Hz (India) or 60 Hz (US)
    # get the power spectrum, then compare power at 50/60 with power just next to it
    freqs, psd = welch(data, fs=sfreq, nperseg=int(2 * sfreq), axis=-1)

    best = {"line_freq": None, "median_ratio": 0.0, "affected_channels": []}

    for line in (50, 60):
        if line >= sfreq / 2:      # sampling rate too low to even see this frequency
            continue
        peak_band = (freqs >= line - 1) & (freqs <= line + 1)       # right at 50/60
        ref_band = ((freqs >= line - 6) & (freqs < line - 3)) | \
                   ((freqs > line + 3) & (freqs <= line + 6))       # just around it

        peak = psd[:, peak_band].mean(axis=1)
        ref = psd[:, ref_band].mean(axis=1)
        # a flat channel has zero power, 0/0 gave NaN and broke everything before
        ratio = peak / np.where(ref > 0, ref, np.nan)

        if np.all(np.isnan(ratio)):
            continue
        median_ratio = float(np.nanmedian(ratio))    # nanmedian skips the NaNs

        if median_ratio > best["median_ratio"]:
            best = {
                "line_freq": line,
                "median_ratio": round(median_ratio, 2),
                "affected_channels": [n for n, r in zip(ch_names, ratio) if r > ratio_thresh],
            }

    if best["median_ratio"] <= ratio_thresh:
        best["line_freq"] = None     # spike isn't big enough, so no line noise
    return best


# check 3: eye blinks

def check_blinks(data, sfreq, ch_names, front=FRONT_CHANNELS, min_amp_uv=60):
    # a blink is a big slow bump on the electrodes above the eyes
    # plan: take front channels, keep only 1-10 Hz, average them, find the tall peaks
    # name match is flexible so "EEG Fp1-REF" also counts as Fp1
    idx = [i for i, n in enumerate(ch_names)
           if any(f.lower() in n.lower() for f in front)]
    if not idx:
        return {"front_channels": [], "n_blinks": None, "note": "no front channels found"}

    sos = butter(2, [1, 10], btype="band", fs=sfreq, output="sos")   # blinks live in 1-10 Hz
    sig = sosfiltfilt(sos, data[idx], axis=1).mean(axis=0)           # average of front channels

    mad = np.median(np.abs(sig - np.median(sig)))
    robust_sigma = 1.4826 * mad                    # normal noise level, blinks don't affect it
    thresh = max(min_amp_uv, 5 * robust_sigma)     # blink must be well above normal

    # distance = 0.4 s gap so one blink doesn't get counted twice
    peaks, _ = find_peaks(np.abs(sig), height=thresh, distance=int(0.4 * sfreq))

    duration = data.shape[1] / sfreq
    seconds_hit = {int(p / sfreq) for p in peaks}  # seconds that have a blink in them

    return {
        "front_channels": [ch_names[i] for i in idx],
        "n_blinks": int(len(peaks)),
        "blinks_per_min": round(len(peaks) / (max(duration, 1) / 60), 1),
        "blink_times_s": [round(float(p / sfreq), 1) for p in peaks],
        "contaminated_pct": round(100 * len(seconds_hit) / max(int(duration), 1), 1),
        "threshold_uv": round(float(thresh), 1),
    }


# overall report

def quality_report(data, sfreq, ch_names):
    # run all 3 checks, then turn them into one score out of 100
    chan = check_flat_noisy(data, sfreq, ch_names)
    line = check_line_noise(data, sfreq, ch_names)
    blink = check_blinks(data, sfreq, ch_names)
    n = len(ch_names)

    flat = [c["channel"] for c in chan if c["status"] == "flat"]
    noisy = [c["channel"] for c in chan if c["status"] == "noisy"]

    # score 1: bad channels. every 1% of bad channels costs 2 points
    frac_bad = (len(flat) + len(noisy)) / n
    channel_score = max(0.0, 100 - 200 * frac_bad)

    # score 2: line noise. hum on the whole recording is worse than a few channels
    frac_aff = len(line["affected_channels"]) / n
    if line["line_freq"]:
        line_score = 100 - 100 * frac_aff
    else:
        line_score = 100 - 50 * frac_aff

    # score 3: blinks. normal and can be removed later, so small penalty
    if blink["n_blinks"] is None:
        blink_score = 100.0          # couldn't check, so don't punish
    else:
        blink_score = max(0.0, 100 - 0.5 * blink["contaminated_pct"])

    # weights: channels 40%, line noise 30%, blinks 30% (my guess, can change later)
    overall = round(0.4 * channel_score + 0.3 * line_score + 0.3 * blink_score, 1)
    label = "Good" if overall >= 85 else "Fair" if overall >= 60 else "Poor"

    # list of problems in plain english, for the app to show later
    issues = []
    if flat:
        issues.append(f"Flat channels: {', '.join(flat)}")
    if noisy:
        issues.append(f"Very noisy channels: {', '.join(noisy)}")
    if line["line_freq"]:
        issues.append(f"{line['line_freq']} Hz power-line noise across the recording")
    elif line["affected_channels"]:
        issues.append("Possible power-line noise on: " + ", ".join(line["affected_channels"]))
    if blink["n_blinks"] is None:
        issues.append("Blinks not checked (no front electrodes found)")
    elif blink["n_blinks"] > 0:
        issues.append(f"{blink['n_blinks']} eye blinks detected ({blink['blinks_per_min']}/min)")
    if not issues:
        issues.append("No problems found")

    return {
        "score": overall,
        "label": label,
        "sub_scores": {
            "channels": round(channel_score, 1),
            "line_noise": round(line_score, 1),
            "blinks": round(blink_score, 1),
        },
        "issues": issues,
        "details": {"channels": chan, "line_noise": line, "blinks": blink},
    }
