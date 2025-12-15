"""load_eeg.py: Module to load EEG data recorded from a hyperscanning experiment

Data is found in the "data/" directory
and consists of 2 recordings per subject:
- 71_{subject_id}_StoryCorps_Q&A.{set,fdt} -> this is the joint storycorps watching
- 72_{subject_id}_BangBangYouAreDead.{set,fdt} -> this is the bang bang you are dead watching

32 channel EEG with a 250Hz sample rate, and 125Hz low-pass
"""
import matplotlib.pyplot as plt
import mne  # type: ignore

def load_eeg(subject_id: int, stimulus: str) -> mne.io.Raw:
    """Load EEG data for a given subject and stimulus.

    Args:
        subject_id (int): The ID of the subject (e.g., 1, 2, ...).
        stimulus (str): The stimulus name, either 'StoryCorps_Q&A' or 'BangBangYouAreDead'.

    Returns:
        mne.io.Raw: The loaded EEG data.
    """
    if stimulus == "StoryCorps_Q&A":
        file_code = 71
    elif stimulus == "BangBangYouAreDead":
        file_code = 72
    else:
        raise ValueError("Invalid stimulus name. Must be 'StoryCorps_Q&A' or 'BangBangYouAreDead'.")
    
    eog_channel_names = ['HL_EOG', 'HR_EOG', 'VA_EOG', 'VB_EOG']

    data_path = "data/{}_HS1{:02d}_{}.set".format(file_code, subject_id, stimulus)
    raw = mne.io.read_raw_eeglab(data_path, preload=True)
    raw.set_channel_types({ch: 'eog' for ch in eog_channel_names})
    return raw

if __name__ == "__main__":
    subject_id = 1
    stimulus = "StoryCorps_Q&A"
    raw_eeg = load_eeg(subject_id, stimulus)
    print(raw_eeg.info)
    plt.ion()
    raw_eeg.compute_psd(fmin=0.01, fmax=50.0).plot(average=True, show=True)
    raw_eeg.plot(scalings='auto', duration=2.0, show=True, block=True, title=f"Subject {subject_id} - {stimulus}")
    plt.ioff()
