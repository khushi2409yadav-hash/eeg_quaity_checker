import os
import tempfile

import mne
import pandas as pd
import streamlit as st

# where mne stores the sample data
mne_folder = os.path.join(os.path.expanduser("~"), "mne_data")
os.makedirs(mne_folder, exist_ok=True)
mne.set_config("MNE_DATASETS_EEGBCI_PATH", mne_folder, set_env=True)

from quality import load_recording, load_eegbci, make_fake_eeg, quality_report

st.set_page_config(page_title="EEG Quality Checker", page_icon="🧠")
st.title("EEG Signal Quality Checker")
st.write("Upload an EEG recording and get a quality report. "
         "No machine learning, just signal checks.")


@st.cache_data
def get_sample():
    # download once, then streamlit remembers it
    return load_eegbci(subject=1, run=1)


source = st.radio(
    "Where should the data come from?",
    ["Upload my own .edf file", "PhysioNet sample recording", "Fake data with problems"],
)

data = None

if source == "Upload my own .edf file":
    f = st.file_uploader("Choose an .edf file", type=["edf"])
    if f is not None:
        # mne needs a file path, so save the upload to a temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(f.read())
            path = tmp.name
        try:
            data, sfreq, ch_names = load_recording(path)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
        finally:
            os.remove(path)

elif source == "PhysioNet sample recording":
    try:
        with st.spinner("Loading sample..."):
            data, sfreq, ch_names = get_sample()
    except Exception as e:
        st.error(f"Could not download the sample right now: {e}")

else:
    data, sfreq, ch_names = make_fake_eeg(
        flat_ch=2, noisy_ch=4, line_hz=50, blink_times=(3, 8, 14, 21, 27)
    )

if data is not None:
    report = quality_report(data, sfreq, ch_names)

    st.header("Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall score", f"{report['score']} / 100")
    c1.caption(report["label"])
    c2.metric("Channels", report["sub_scores"]["channels"])
    c3.metric("Line noise", report["sub_scores"]["line_noise"])
    c4.metric("Blinks", report["sub_scores"]["blinks"])

    st.subheader("Problems found")
    for issue in report["issues"]:
        st.write("•", issue)

    st.subheader("Channel details")
    st.dataframe(pd.DataFrame(report["details"]["channels"]), use_container_width=True)

    st.caption(f"{len(ch_names)} channels, {sfreq:g} Hz, "
               f"{data.shape[1] / sfreq:.0f} seconds")
