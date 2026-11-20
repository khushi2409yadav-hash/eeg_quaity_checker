# EEG Quality Checker

A small web app that checks whether an EEG recording is clean enough to use. You give it a recording, and it tells you which parts look bad and gives an overall quality score out of 100.

Live demo: https://eegquaitychecker-tdsjdovb6qvpkxpaql4bmn.streamlit.app/

I made this because I kept running into noisy EEG data in my other projects, and I wanted a quick way to check a recording before analysing it. There is no machine learning here, only simple signal checks.

## What it checks

- **Flat or very noisy channels:** an electrode that is dead or picking up way too much noise.
- **Power-line noise:** extra power at 50/60 Hz from the electrical supply.
- **Eye blinks:** big spikes on the front electrodes, which are usually blinks and not brain activity.

Each check feeds into one overall score, and the app labels the recording (for example "Good") based on it.

## How to use it

Open the live demo and either upload a recording or use one of the built-in options:

- a fake EEG generator, where you can see how the score changes when noise is added
- a sample recording from PhysioNet (EEGBCI, subject 1, run 1)

## Running it on your computer

```
git clone
