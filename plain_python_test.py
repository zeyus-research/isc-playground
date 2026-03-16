# %% [markdown]
# Reference GSR: (but from shortened - 7 minute video)

# %% [markdown]
# 
# ...
# 
# 

# %% [markdown]
# 

# %%
# optional MNE cuda settings (comment out if you dont have CUDA or ROCm)

from mne import set_config
set_config('MNE_USE_CUDA', 'true', set_env=True)

# for rocm7:
# uv run pip install cupy-rocm-7-0 --pre -U -f https://pip.cupy.dev/pre

# for other ROCm versions:
# these should be set in a .env file assuming you use direnv,
# otherwise, set them however you want but they will need
# to be configured for cupy installation if you want to use ROCm
# set_config('CUPY_INSTALL_USE_HIP', '1', set_env=True)
# set_config('HCC_AMDGPU_TARGET', 'gfx1102', set_env=True)
# set_config('ROCM_HOME', '/opt/rocm', set_env=True)
# set_config('__HIP_PLATFORM_HCC__', '', set_env=True)

# it is also possible to specify an env file when you install cupy:
# uv run --env-file .env -- pip install cupy 

# now cuda can be initalized by MNE-Python
import mne
mne.cuda.init_cuda()

# %%
# imports
import io
from copy import copy
from collections import OrderedDict
import tempfile  
import numpy as np
import scipy
from scipy import linalg as sp_linalg
import matplotlib.pyplot as plt
from pathlib import Path

import mne

from hypyp import prep 
from hypyp import analyses
from hypyp import stats as hyp_stats
from hypyp import viz
from ipyfilechooser import FileChooser
from IPython.display import display


# %%
data_path: Path = ""

# check env variable for default data path
import os
env_data_path = os.getenv('EEG_DATA_PATH')
if env_data_path is not None:
    data_path = Path(env_data_path)

fc = FileChooser(str(data_path))
fc.title = "Select EEG data directory"
fc.show_only_dirs = True
display(fc)

def on_submit(chooser):
    global data_path
    data_path = Path(chooser.selected)
    print(f"Data path set to: {data_path}")

fc.register_callback(on_submit)



# %%
from mne_icalabel import label_components
import copy

def ICA_autocorrect(icas: list, epochs: list, verbose: bool = False) -> list:
    """
    Automatically detect the ICA components that are not brain related and remove them.

    Arguments:
        icas: list of Independent Components for each participant (IC are MNE
          objects).
        epochs: list of 2 Epochs objects (for each participant). Epochs_S1
          and Epochs_S2 correspond to a condition and can result from the
          concatenation of Epochs from different experimental realisations
          of the condition.
          Epochs are MNE objects: data are stored in an array of shape
          (n_epochs, n_channels, n_times) and parameters information is
          stored in a disctionnary.
        verbose: option to plot data before and after ICA correction, 
          boolean, set to False by default. 

    Returns:
        cleaned_epochs_ICA: list of 2 cleaned Epochs for each participant
          (the non-brain related IC have been removed from the signal).
    """

    cleaned_epochs_ICA = []
    for ica, epoch in zip(icas, epochs):
        ica_with_labels_fitted = label_components(epoch, ica, method="iclabel")
        ica_with_labels_component_detected = ica_with_labels_fitted["labels"]
        # Remove non-brain components (take only brain components for each subject)
        excluded_idx_components = [idx for idx, label in enumerate(ica_with_labels_component_detected) if label not in ["brain"]]
        cleaned_epoch_ICA = mne.Epochs.copy(epoch)
        cleaned_epoch_ICA.info['bads'] = []
        ica.apply(cleaned_epoch_ICA, exclude=excluded_idx_components)
        cleaned_epoch_ICA.info['bads'] = copy.deepcopy(epoch.info['bads'])
        cleaned_epochs_ICA.append(cleaned_epoch_ICA)

        if verbose:
            epoch.plot(title='Before ICA correction', show=True)
            cleaned_epoch_ICA.plot(title='After ICA correction',show=True)
    return cleaned_epochs_ICA


# %%
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
    
    # scalp_channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'P3', 'P4', 'P7', 'P8', 'T7', 'T8', 'AFz']

    # Ear-EEG - exclude from 10-20 montage
    ear_eeg_channels = ['ELA1', 'ELA2', 'ELB1', 'ELB2', 'ELC1', 'ELC2', 'ELK', 'ELT',
                        'ERA1', 'ERA2', 'ERB1', 'ERB2', 'ERC1', 'ERC2', 'ERK', 'ERT']

    # EOG - set channel type
    eog_channels = ['HL_EOG', 'HR_EOG', 'VA_EOG', 'VB_EOG']

    file_path = data_path / "{}_HS1{:02d}_{}.set".format(file_code, subject_id, stimulus)
    raw = mne.io.read_raw_eeglab(file_path, preload=True, eog=eog_channels, verbose=False)

    # After loading your raw data:
    raw.set_channel_types({ch: 'eog' for ch in eog_channels})
    raw.set_channel_types({ch: 'misc' for ch in ear_eeg_channels})  # or 'eeg' if you want to keep them

    # Apply standard montage to just the scalp channels
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')  # ignores ear-EEG and EOG

    
    return raw

# %% [markdown]
# # Parra et al. CorrCA


# %%
# ICA https://github.com/ML-D00M/ISC-Inter-Subject-Correlations/blob/main/Python/ISC.py
from scipy.linalg import eigh

def train_cca(data):
    """Run Correlated Component Analysis on your training data.

    Parameters:
    ----------
    data : dict
        Dictionary with keys are names of conditions and values are numpy
        arrays structured like (subjects, channels, samples).
        The number of channels must be the same between all conditions!

    Returns:
    -------
    W : np.array
        Columns are spatial filters. They are sorted in descending order, it means that first column-vector maximize
        correlation the most.
    ISC : np.array
        Inter-subject correlation sorted in descending order

    """

    # start = default_timer()

    C = len(data.keys())
    # st.write(f"train_cca - calculations started. There are {C} conditions")

    gamma = 0.1
    Rw, Rb = 0, 0
    for c, cond in data.items():
        (
            N,
            D,
            T,
        ) = cond.shape
        # st.write(f"Condition '{c}' has {N} subjects, {D} sensors and {T} samples")
        cond = cond.reshape(D * N, T)

        # Rij
        Rij = np.swapaxes(np.reshape(np.cov(cond), (N, D, N, D)), 1, 2)

        # Rw
        Rw = Rw + np.mean([Rij[i, i, :, :] for i in range(0, N)], axis=0)

        # Rb
        Rb = Rb + np.mean(
            [Rij[i, j, :, :] for i in range(0, N) for j in range(0, N) if i != j],
            axis=0,
        )

    # Divide by number of condition
    Rw, Rb = Rw / C, Rb / C

    # Regularization
    Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # ISCs and Ws
    [ISC, W] = eigh(Rb, Rw_reg)

    # Make descending order
    ISC, W = ISC[::-1], W[:, ::-1]

    # stop = default_timer()

    # st.write(f"Elapsed time: {round(stop - start)} seconds.")
    return W, ISC


def apply_cca(X, W, fs):
    """Applying precomputed spatial filters to your data.

    Parameters:
    ----------
    X : ndarray
        3-D numpy array structured like (subject, channel, sample)
    W : ndarray
        Spatial filters.
    fs : int
        Frequency sampling.
    Returns:
    -------
    ISC : ndarray
        Inter-subject correlations values are sorted in descending order.
    ISC_persecond : ndarray
        Inter-subject correlations values per second where first row is the most correlated.
    ISC_bysubject : ndarray
        Description goes here.
    A : ndarray
        Scalp projections of ISC.
    """

    # start = default_timer()
    # st.write("apply_cca - calculations started")

    N, D, T = X.shape
    # gamma = 0.1
    window_sec = 5
    X = X.reshape(D * N, T)

    # Rij
    Rij = np.swapaxes(np.reshape(np.cov(X), (N, D, N, D)), 1, 2)

    # Rw
    Rw = np.mean([Rij[i, i, :, :] for i in range(0, N)], axis=0)
    # Rw_reg = (1 - gamma) * Rw + gamma * np.mean(eigh(Rw)[0]) * np.identity(Rw.shape[0])

    # Rb
    Rb = np.mean(
        [Rij[i, j, :, :] for i in range(0, N) for j in range(0, N) if i != j], axis=0
    )

    # ISCs
    ISC = np.sort(
        np.diag(np.transpose(W) @ Rb @ W) / np.diag(np.transpose(W) @ Rw @ W)
    )[::-1]

    # Scalp projections
    A = np.linalg.solve((np.transpose(W) @ Rw @ W).T, (Rw @ W).T).T

    # ISC by subject
    # st.write("by subject is calculating")
    ISC_bysubject = np.empty((D, N))

    for subj_k in range(0, N):
        Rw, Rb = 0, 0
        Rw = np.mean(
            [
                Rw
                + 1 / (N - 1) * (Rij[subj_k, subj_k, :, :] + Rij[subj_l, subj_l, :, :])
                for subj_l in range(0, N)
                if subj_k != subj_l
            ],
            axis=0,
        )
        Rb = np.mean(
            [
                Rb
                + 1 / (N - 1) * (Rij[subj_k, subj_l, :, :] + Rij[subj_l, subj_k, :, :])
                for subj_l in range(0, N)
                if subj_k != subj_l
            ],
            axis=0,
        )

        ISC_bysubject[:, subj_k] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(
            np.transpose(W) @ Rw @ W
        )

    # ISC per second
    # st.write("by persecond is calculating")
    ISC_persecond = np.empty((D, int(T / fs) + 1))
    window_i = 0

    for t in range(0, T, fs):
        t_end = min(t + window_sec * fs, T)
        if t_end - t < 1:
            break
        Xt = X[:, t : t_end]
        Rij = np.cov(Xt)
        Rw = np.mean([Rij[i : i + D, i : i + D] for i in range(0, D * N, D)], axis=0)
        Rb = np.mean(
            [
                Rij[i : i + D, j : j + D]
                for i in range(0, D * N, D)
                for j in range(0, D * N, D)
                if i != j
            ],
            axis=0,
        )

        ISC_persecond[:, window_i] = np.diag(np.transpose(W) @ Rb @ W) / np.diag(
            np.transpose(W) @ Rw @ W
        )
        window_i += 1

    # stop = default_timer()
    # st.write(f"Elapsed time: {round(stop - start)} seconds.")

    return ISC, ISC_persecond, ISC_bysubject, A

# %% [markdown]
# # Setup

# %%

# Define frequency bands as a dictionary
freq_bands = {
    'Alpha-Low': [7.5, 11],
    'Alpha-High': [11.5, 13]
}

# Convert to an OrderedDict to keep the defined order
freq_bands = OrderedDict(freq_bands)
print('Frequency bands:', freq_bands)


# %%
# Define important movie events (in seconds)
byd_movie_events = {
    'gun_mother': 441,
    'gun_mother_father': 471,
    'gun_postman': 505,
    'gun_loading': 664,
    'next_bullet': 746,
    'gun_girl': 13*60 + 34,
    'more_loading': 14*60 + 7,
    'even_more_loading': 15*60 + 57,
    'gun_shopkeeper': 16*60 + 54,
    'gun_maid': 20*60 + 17,
    'gun_fire': 20*60 + 25,
}

subject_ids = [i for i in range(1, 11)]  # Subject IDs from 1 to 10


# %% [markdown]
# # Add events to raw data

# %%
# load data
raws = {}
for subject_id in subject_ids:
    raws[subject_id] = load_eeg(subject_id, stimulus='BangBangYouAreDead')
    print(f'Loaded EEG data for Subject {subject_id}')
    # check for NaN values
    if np.any(np.isnan(raws[subject_id].get_data())):
        print(f'Subject {subject_id} has NaN values in the data.')
        # show count
        nan_count = np.isnan(raws[subject_id].get_data()).sum()
        print(f'Number of NaN values: {nan_count}')
        # show information about channels that contain NaNs
        for ch_idx, ch_name in enumerate(raws[subject_id].ch_names):
            ch_data = raws[subject_id].get_data(picks=[ch_idx])
            if np.any(np.isnan(ch_data)):
                ch_nan_count = np.isnan(ch_data).sum()
                print(f'  Channel {ch_name} has {ch_nan_count} NaN values, marking as bad.')
                raws[subject_id].info['bads'].append(ch_name)
                # if it's an EEG channel, mark it as bad and print a prominent warning
                if raws[subject_id].get_channel_types(picks=[ch_idx])[0] == 'eeg':
                    print(f'======================                    WARNING: Marked EEG channel {ch_name} as bad due to NaN values.                     ======================')  




# %%
def add_movie_events_to_raw(raw, movie_events_dict):
    """
    Add movie events as annotations to raw EEG data.
    
    Args:
        raw: MNE Raw object
        movie_events_dict: Dictionary with event names and onset times in seconds
        
    Returns:
        raw: Raw object with annotations added
    """
    # Create annotations for each movie event
    onsets = []
    durations = []
    descriptions = []
    
    for event_name, onset_time in movie_events_dict.items():
        onsets.append(onset_time)
        durations.append(0)  # Point events
        descriptions.append(event_name)
    
    # Create Annotations object
    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=raw.info['meas_date']
    )
    
    # Add annotations to raw data
    raw.set_annotations(raw.annotations + annotations)
    
    return raw

# %%
# Add movie events to all preprocessed raw objects
for subject_id, preprocessed_raw in raws.items():
    raws[subject_id] = add_movie_events_to_raw(
        preprocessed_raw, 
        byd_movie_events
    )
    print(f'Added movie events to Subject {subject_id}')

# Verify annotations were added (check first subject)
print("\nAnnotations for Subject 1:")
print(raws[1].annotations)

# %%
def create_epochs_from_movie_events(raw, event_names=None, tmin=-2.0, tmax=5.0, baseline=None):
    """
    Create epochs around movie events.
    
    Args:
        raw: MNE Raw object with annotations
        event_names: List of event names to epoch on. If None, uses all annotations.
        tmin: Start time before event (in seconds)
        tmax: End time after event (in seconds)
        baseline: Baseline correction tuple (start, end) in seconds, or None
        
    Returns:
        epochs: MNE Epochs object
    """
    # Convert annotations to events
    events, event_id = mne.events_from_annotations(raw)
    
    # Filter event_id if specific event names are provided
    if event_names is not None:
        event_id_filtered = {k: v for k, v in event_id.items() if k in event_names}
    else:
        event_id_filtered = event_id
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id_filtered,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=True
    )
    
    return epochs

# %% [markdown]
# # Create Epochs Around High Tension Events
# 
# Now we'll create epochs around all the movie events (gun scenes, loading scenes, etc.)
# 
# # TODO:
# 
# try: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00309/full
# 

# %%
# Create epochs for all subjects
# Time window: 2 seconds before to 5 seconds after each event
epochs_dict = {}

montage = mne.channels.make_standard_montage('standard_1020')
# trim all files to the same length
min_length = min([raws[s].n_times for s in raws])
raws_filtered_no_regres: dict[int, mne.io.Raw] = {}
raws_filtered: dict[int, mne.io.Raw] = {}
for subject_id, raw in raws.items():
    raw_filtered: mne.io.Raw = raw.copy()
    raw_filtered.crop(tmax=(min_length - 1) / raw.info['sfreq'])
    raw_filtered.set_montage(montage)
    # butterworth high-pass order 5 (0.5 Hz)
    raw_filtered.filter(l_freq=0.5, h_freq=40, method='fir', picks='all')

    # save copy before dropping misc channels and regressing
    raws_filtered_no_regres[subject_id] = raw_filtered.copy()
    # drop misc channels (ear-EEG) before regression — NaN in misc channels
    # propagates through SSP projection via 0 * NaN = NaN in numpy
    raw_filtered.pick(picks=['eeg', 'eog'])
    # regress out EOG channels
    raw_filtered.set_eeg_reference('average', projection=False)
    weights = mne.preprocessing.EOGRegression().fit(raw_filtered)
    raw_filtered = weights.apply(raw_filtered) # no need to keep EOG channels anymore
    raw_filtered.pick(picks='eeg')
    eeg_channel_indices = mne.pick_types(raw_filtered.info, eeg=True, eog=False, meg=False, exclude='bads')

    # detect outliers > 4 IQD
    percentiles = np.percentile(abs(raw_filtered.get_data()), [25, 75], axis=1)
    # replace bad samples and 40ms before and after with 0s
    for ch_idx in eeg_channel_indices:
        iqd = percentiles[1, ch_idx] - percentiles[0, ch_idx]
        threshold = percentiles[1, ch_idx] + 4 * iqd
        data = raw_filtered.get_data(picks = [ch_idx])[0]
        bad_samples = np.where(abs(data) > threshold)[0]
        for sample in bad_samples:
            start = max(0, sample - int(0.04 * raw_filtered.info['sfreq']))
            end = min(len(data), sample + int(0.04 * raw_filtered.info['sfreq']))
            data[start:end] = 0
        raw_filtered._data[ch_idx, :] = data
    
    # find bad channels base on power outliers
    log_power = np.log(np.std(raw_filtered.get_data(), axis=1))
    power_threshold = np.percentile(np.log(np.std(raw_filtered.get_data(), axis=1)), [25, 50, 75])
    bad_channel_indices = np.where(log_power > power_threshold[2] + 4 * (power_threshold[2] - power_threshold[0]))[0]
    bad_channel_names = [raw_filtered.ch_names[idx] for idx in bad_channel_indices]
    raw_filtered.info['bads'].extend(bad_channel_names)
    print(f'Subject {subject_id}: Marked bad channels: {bad_channel_names}')
    


    epochs = create_epochs_from_movie_events(
        # raws_filtered_no_regres[subject_id],
        raw_filtered,
        event_names=byd_movie_events,  # Use all movie events, or specify a list like ['gun_mother', 'gun_fire']
        tmin=-2.0,  # 2 seconds before event
        tmax=5.0,   # 5 seconds after event
    )
    epochs_dict[subject_id] = epochs
    raws_filtered[subject_id] = raw_filtered
    print(f'Created {len(epochs)} epochs for Subject {subject_id}')

# Display info about epochs for first subject
print("\nEpoch info for Subject 1:")
print(epochs_dict[1])

# %%
from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
parula_map_r = LinearSegmentedColormap.from_list('parula_r', cm_data[::-1])
# For use of "viscm view"
test_cm = parula_map

# %% [markdown]
# # ISC
# 
# 1. Calculate ISC for entire recording (not epoched)
# 2. Show overall and windowed ISC results
# 3. calculate ISC on epoched data
# 

# %%
# figure out why some subject data is all NaN.
for subject_id, raw in raws_filtered.items():
    subj_data = raw.get_data(picks='eeg')
    if np.any(np.isnan(subj_data)):
        print(f'Subject {subject_id} has NaN values in the data.')
        # show count
        nan_count = np.isnan(subj_data).sum()
        print(f'Number of NaN values: {nan_count}')
        print(f'Subject {subject_id} data shape: {subj_data.shape}')
        #print channel names with NaNs
        for ch_idx, ch_name in enumerate(raw.ch_names):
            ch_data = raw.get_data(picks=[ch_idx])
            if np.any(np.isnan(ch_data)):
                ch_nan_count = np.isnan(ch_data).sum()
                print(f'  Channel {ch_name} has {ch_nan_count} NaN values.')
    

# %%
# calculate ISC for recording
# format data based on expectation:
# Dictionary with keys are names of conditions and values are numpy
# arrays structured like (subjects, channels, samples)

# first full recordings
data_full = {}
for subject_id in subject_ids:
    raw = raws_filtered[subject_id]
    data_full[subject_id] = raw.get_data(picks='eeg') 
data_full_array = np.array([data_full[s] for s in subject_ids])



# start_sample = data_full_array[0].shape[1] - int(197 * int(raws_filtered[subject_ids[0]].info['sfreq']))
# data_full_array = data_full_array[:, :, start_sample:]
# print(f'Full recording data shape (subjects, channels, samples): {data_full_array.shape}')

W, ISC = train_cca({'full_recording': data_full_array})

# apply CCA
ISC_a, ISC_persecond, ISC_bysubject, A = apply_cca(data_full_array, W, fs=int(raws_filtered[subject_ids[0]].info['sfreq']))
print('Calculated ISC for full recordings')

# find maximumally correlated time windows for component 1
def find_max_isc_windows(isc_persecond, window_size_sec=5, top_n=3, component=0):
    """
    Find the time windows with the highest ISC values.
    Args:
        isc_persecond: ISC values per second (components x time)
        window_size_sec: Size of the time window in seconds
        top_n: Number of top windows to return
    Returns:
        List of tuples (start_time, end_time, mean_isc) for the top_n windows
    """
    n_components, n_timepoints = isc_persecond.shape
    half_window = window_size_sec // 2
    mean_isc_values = []

    for t in range(half_window, n_timepoints - half_window):
        window_isc = isc_persecond[component, t - half_window:t + half_window]
        mean_isc = np.mean(window_isc)
        mean_isc_values.append((t - half_window, t + half_window, mean_isc))

    # Sort by mean ISC and get top_n windows
    mean_isc_values.sort(key=lambda x: x[2], reverse=True)
    top_windows = mean_isc_values[:top_n]

    return top_windows

# get top 3 windows for component 1
top_windows = find_max_isc_windows(ISC_persecond, window_size_sec=5, top_n=3, component=0)
for start, end, mean_isc in top_windows:
    print(f'Top ISC window for component 1: {start}s to {end}s with mean ISC: {mean_isc}')

# get top 3 windows for component 2
top_windows = find_max_isc_windows(ISC_persecond, window_size_sec=5, top_n=3, component=1)
for start, end, mean_isc in top_windows:
    print(f'Top ISC window for component 2: {start}s to {end}s with mean ISC: {mean_isc}')


# get top 3 windows for component 3
top_windows = find_max_isc_windows(ISC_persecond, window_size_sec=5, top_n=3, component=2)
for start, end, mean_isc in top_windows:
    print(f'Top ISC window for component 3: {start}s to {end}s with mean ISC: {mean_isc}')



def find_max_isc_window_timerange(isc_persecond, window_size_sec=5, time_range_s=197, component=0):
    """Finds the time range with the highest sum of windowed ISC values.
        i.e. max sequential window of windows in the entire recording
    """
    n_components, n_timepoints = isc_persecond.shape
    half_window = window_size_sec // 2
    n_windows_in_range = time_range_s // window_size_sec
    max_sum_isc = -np.inf
    best_start = 0

    for t in range(half_window, n_timepoints - half_window - n_windows_in_range + 1):
        sum_isc = 0
        for w in range(n_windows_in_range):
            window_start = t + w * window_size_sec - half_window
            window_end = t + w * window_size_sec + half_window
            window_isc = isc_persecond[component, window_start:window_end]
            sum_isc += np.mean(window_isc)

        if sum_isc > max_sum_isc:
            max_sum_isc = sum_isc
            best_start = t

    best_end = best_start + n_windows_in_range * window_size_sec
    return best_start, best_end, max_sum_isc

best_start, best_end, max_isc = find_max_isc_window_timerange(ISC_persecond, window_size_sec=5, time_range_s=197, component=0)
print(f'Max correlated 197s time range for component 1: {best_start}s to {best_end}s with mean ISC: {max_isc / (197 // 5)}')

best_start, best_end, max_isc = find_max_isc_window_timerange(ISC_persecond, window_size_sec=5, time_range_s=197, component=1)
print(f'Max correlated 197s time range for component 2: {best_start}s to {best_end}s with mean ISC: {max_isc / (197 // 5)}')

best_start, best_end, max_isc = find_max_isc_window_timerange(ISC_persecond, window_size_sec=5, time_range_s=197, component=2)
print(f'Max correlated 197s time range for component 3: {best_start}s to {best_end}s with mean ISC: {max_isc / (197 // 5)}')

# start_sample = data_full_array[0].shape[1] - int(197 * int(raws_filtered[subject_ids[0]].info['sfreq']))
# data_full_array = data_full_array[:, :, start_sample:]
# print(f'Full recording data shape (subjects, channels, samples): {data_full_array.shape}')

start_sample = 1197 * int(raws_filtered[subject_ids[0]].info['sfreq'])
end_sample = (1197+197) * int(raws_filtered[subject_ids[0]].info['sfreq'])
data_segment_array = data_full_array[:, :, start_sample:end_sample]
print(f'Segment data shape (subjects, channels, samples): {data_segment_array.shape}')

W_seg, ISC_seg = train_cca({'segment_recording': data_segment_array})
ISC_a_seg, ISC_persecond_seg, ISC_bysubject_seg, A_seg = apply_cca(data_segment_array, W_seg, fs=int(raws_filtered[subject_ids[0]].info['sfreq']))
print('Calculated ISC for segment recordings')




isc_all = {'full_recording': {
        'ISC': ISC_a,
        'ISC_persecond': ISC_persecond,
        'ISC_bysubject': ISC_bysubject,
        'A': A
    }}


isc_all_seg = {'segment_recording': {
        'ISC': ISC_a_seg,
        'ISC_persecond': ISC_persecond_seg,
        'ISC_bysubject': ISC_bysubject_seg,
        'A': A_seg
    }}







# %%
def plot_isc(isc_all):
    # plot ISC as a bar chart
    plot1 = plt.figure()
    # get number of components
    n_comp = len(isc_all[list(isc_all.keys())[0]]["ISC"])
    comp1 = [cond["ISC"][0] for cond in isc_all.values()]
    if n_comp > 1:
        comp2 = [cond["ISC"][1] for cond in isc_all.values()]
    if n_comp > 2:
        comp3 = [cond["ISC"][2] for cond in isc_all.values()]
    barWidth = 0.2
    r1 = np.arange(len(comp1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, comp1, color="red", width=barWidth, edgecolor="white", label="Comp1")
    if n_comp > 1:
        plt.bar(
            r2, comp2, color="purple", width=barWidth, edgecolor="white", label="Comp2"
        )
    if n_comp > 2:
        plt.bar(
            r3, comp3, color="blue", width=barWidth, edgecolor="white", label="Comp3"
        )
    plt.xticks([r + barWidth for r in range(len(comp1))], isc_all.keys())
    plt.ylabel("ISC", fontweight="bold")
    plt.title("ISC for each condition")
    plt.legend()
    plt.tight_layout()

    return plot1
def plot_isc_time(isc_all):
    plot = plt.figure()
    # plot ISC_persecond
    n_comp = len(isc_all[list(isc_all.keys())[0]]["ISC"])
    for cond in isc_all.values():
        for comp_i in range(0, min(n_comp, 3)):
            plt.subplot(3, 1, comp_i + 1)
            plt.title(f"Component {comp_i + 1}", loc="right")
            plt.plot(cond["ISC_persecond"][comp_i], marker="o", linestyle="-", markersize=3)
            plt.ylim(0, 1)
    plt.tight_layout()
    # add space at left and bottom
    plot.subplots_adjust(left=0.12, bottom=0.12)

    # make a shared x-axis
    plot.text(0.5, 0.04, "Time (s)", ha="center")

    # and a shared y-axis
    plot.text(0.04, 0.5, "ISC", va="center", rotation="vertical")

    # add a single legend for all subplots
    plt.figlegend(isc_all.keys(), loc="upper left")


    return plot

def plot_isc_subj(isc_all):
    plot = plt.figure()
    # plot ISC_bysubject
    n_comp = len(isc_all[list(isc_all.keys())[0]]["ISC"])
    for cond in isc_all.values():
        for comp_i in range(0, min(n_comp, 3)):
            plt.subplot(3, 1, comp_i + 1)
            plt.grid(visible=True, which="both", axis="y")
            plt.title(f"Component {comp_i + 1}", loc="right")
            plt.plot(cond["ISC_bysubject"][comp_i], marker="o", linestyle="")
            plt.ylim(0, 1)
    plt.tight_layout()
    # add space at left and bottom
    plot.subplots_adjust(left=0.12, bottom=0.12)

    # make a shared x-axis
    plot.text(0.5, 0.04, "Subject", ha="center")
    # and a shared y-axis
    plot.text(0.04, 0.5, "ISC", va="center", rotation="vertical")

    # add a single legend for all subplots
    plt.figlegend(isc_all.keys(), loc="upper left")

    
    return plot


plot_isc_time(isc_all)
plot_isc(isc_all)
plot_isc_subj(isc_all)

plot_isc_time(isc_all_seg)
plot_isc(isc_all_seg)
plot_isc(isc_all_seg)




# %%
# now we can do the same to the epoched data, the train_cca function expects the dictionary keys to be condition names (in our case, event names), and the values to be numpy arrays of shape (subjects, channels, samples)
data_epochs = {}
for event_name in byd_movie_events.keys():
    # collect epochs for this event across subjects
    epochs_list = []
    for subject_id in subject_ids:
        epochs = epochs_dict[subject_id]
        if event_name in epochs.event_id:
            epoch_data = epochs[event_name].get_data()  # shape (n_epochs, n_channels, n_times)
            # average across epochs for this event
            epoch_data_mean = np.mean(epoch_data, axis=0)  # shape (n_channels, n_times)
            epochs_list.append(epoch_data_mean)
        else:
            print(f'Warning: Event {event_name} not found for Subject {subject_id}')
    # convert to numpy array of shape (subjects, channels, samples)
    data_epochs[event_name] = np.array(epochs_list)
    print(f'Collected data for event {event_name}, shape: {data_epochs[event_name].shape}')
print('Prepared epoched data for all events.')

# run ica on epoched data
W_epochs, ISC_epochs = train_cca(data_epochs)
print('Trained CCA on epoched data.')
# apply CCA
isc_all_epochs = {}
for event_name, data in data_epochs.items():
    ISC_a_epoch, ISC_persecond_epoch, ISC_bysubject_epoch, A_epoch = apply_cca(
        data, W_epochs, fs=int(raws_filtered[subject_ids[0]].info['sfreq'])
    )
    isc_all_epochs[event_name] = {
        'ISC': ISC_a_epoch,
        'ISC_persecond': ISC_persecond_epoch,
        'ISC_bysubject': ISC_bysubject_epoch,
        'A': A_epoch
    }
    print(f'Applied CCA to event {event_name}.')

# plot ISC for epoched data
plot_isc_time(isc_all_epochs)
plot_isc(isc_all_epochs)
plot_isc_subj(isc_all_epochs)

# %% [markdown]
# ## Topography of ISC components

# %%
# plot scalp projections for first 3 components of epoched data
plt.ioff()
for event_name, isc_data in isc_all_epochs.items():
    A_epoch = isc_data['A']
    fig = plt.figure()

    # make subplots for each component
    
    for comp_i in range(3):
        ax = fig.add_subplot(1, 3, comp_i + 1)
        topo_plot = mne.viz.plot_topomap(
            A_epoch[:, comp_i],
            raws_filtered[subject_ids[0]].info,
            cmap=parula_map,
            show=False,
            contours=0,
            axes=ax,
            sphere=0.09,
            extrapolate='head',
            vlim=(np.min(A_epoch[:, comp_i]), np.max(A_epoch[:, comp_i])),
        )
        print(f'Plotted scalp projection for component {comp_i + 1} of event {event_name} with data range ({np.min(A_epoch[:, comp_i])}, {np.max(A_epoch[:, comp_i])})')
        ax.set_title(f'Component {comp_i + 1}')
    # add colorbar to figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(topo_plot[0], cax=cbar_ax)
    # add title
    plt.suptitle(f'Scalp Projections for Event: {event_name}')
    plt.show()
    plt.close(fig)




# %% [markdown]
# # Previous ISC

# %%
# https://github.com/renzocom/CorrCA/blob/master/CorrCA.py
# Renzo Comolatti (renzo.com@gmail.com)
#
# Class with Correlated Component Analysis (CorrCA) method based on
# original matlab code from Parra's lab (https://www.parralab.org/corrca/).
#
# started 18/10/2019

def calc_corrca(epochs, times, **par):
    """
    Calculate Correlated Component Analysis (CorrCA) on given epochs and times.

    Parameters
    ----------
    epochs : ndarray of shape (n_epochs, n_channels, n_times)
        Input signal data.
    times : ndarray of shape (n_times,)
        Array of time points corresponding to the epochs.
    **par : dict
        Additional parameters for the analysis. Expected keys are:
        - 'response_window' : tuple of float
            Start and end time for the response window.
        - 'gamma' : float
            Regularization parameter for the within-subject covariance matrix.
        - 'K' : int
            Number of components to retain.
        - 'n_surrogates' : int
            Number of surrogate datasets to use for statistical testing.
        - 'alpha' : float
            Significance level for statistical testing.
        - 'stats' : bool
            Whether to calculate statistics.

    Returns
    -------
    W : ndarray of shape (n_channels, n_components)
        Backward model (signal to components).
    ISC : ndarray of shape (n_components,)
        Inter-subject correlation values.
    A : ndarray of shape (n_channels, n_components)
        Forward model (components to signal).
    Y : ndarray of shape (n_epochs, n_components, n_times)
        Transformed signal within the response window.
    Yfull : ndarray of shape (n_epochs, n_components, n_times)
        Transformed signal for the entire epoch duration.
    ISC_thr : float
        Threshold for inter-subject correlation values based on surrogate data.
    """
    ini_ix = time2ix(times, par['response_window'][0])
    end_ix = time2ix(times, par['response_window'][1])
    X = np.array(epochs)[..., ini_ix : end_ix]

    W, ISC, A = fit(X, gamma=par['gamma'], k=par['K'])

    n_components = W.shape[1]
    if stats:
        print('Calculating statistics...')
        ISC_thr, ISC_null = stats(X, par['gamma'], par['K'], par['n_surrogates'], par['alpha'])
        n_components = sum(ISC > ISC_thr)
        W, ISC, A = W[:, :n_components], ISC[:n_components], A[:, :n_components]
        
    Y = transform(X, W)
    Yfull = transform(np.array(epochs), W)
    return W, ISC, A, Y, Yfull, ISC_thr

##################
# MAIN FUNCTIONS #
##################
def fit(X, version=2, gamma=0, k=None):
    '''
    Correlated Component Analysis (CorrCA).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal to calculate CorrCA.
    k : int,
        Truncates eigenvalues on the Kth component.
    gamma : float,
        Truncates eigenvalues using SVD.

    Returns
    -------
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).
    ISC : list of floats
        Inter-subject Correlation values.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).
    '''

    # TODO: implement case 3, tsvd truncation

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    if k is not None: # truncate eigenvalues using SVD
        gamma = 0
    else:
        k = D

    # Compute within- (Rw) and between-subject (Rb) covariances
    if False: # Intuitive but innefficient way to calculate Rb and Rw
        Xcat = X.reshape((N * D, T)) # T x (D + N) note: dimensions vary first, then subjects
        Rkl = np.cov(Xcat).reshape((N, D, N, D)).swapaxes(1, 2)
        Rw = Rkl[range(N), range(N), ...].sum(axis=0) # Sum within subject covariances
        Rt = Rkl.reshape(N*N, D, D).sum(axis=0)
        Rb = (Rt - Rw) / (N-1)

    # Rw = sum(np.cov(X[n,...]) for n in range(N))
    # Rt = N**2 * np.cov(X.mean(axis=0))
    # Rb = (Rt - Rw) / (N-1)

    # fix for channel specific bad trial
    temp = [np.cov(X[n,...]) for n in range(N)]
    Rw = np.nansum(temp, axis=0)
    
    Rt = N**2 * np.cov(np.nanmean(X, axis=0))
    
    Rb = (Rt - Rw) / (N-1)
    
    rank = np.linalg.matrix_rank(Rw)
    if rank < D and gamma != 0:
        print('Warning: data is rank deficient (gamma not used).')

    k = min(k, rank) # handle rank deficient data.
    if k < D:
        def regInv(R, k):
            '''PCA regularized inverse of square symmetric positive definite matrix R.'''

            U, S, Vh = np.linalg.svd(R)
            invR = U[:, :k].dot(np.diag(1 / S[:k])).dot(Vh[:k, :])
            return invR

        invR = regInv(Rw, k)
        ISC, W = sp_linalg.eig(invR.dot(Rb))
        ISC, W = ISC[:k], W[:, :k]

    else:
        Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
        ISC, W = sp_linalg.eig(Rb, Rw_reg) # W is already sorted by eigenvalue and normalized

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))

    ISC, W = np.real(ISC), np.real(W)

    if k==D:
        A = Rw.dot(W).dot(sp_linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))

    return W, ISC, A

def transform(X, W):
    '''
    Get CorrCA components from signal(X), e.g. epochs or evoked, using backward model (W).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times) or (n_dim, n_times)
        Signal  to transform.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    '''

    flag = False
    if X.ndim == 2:
        flag = True
        X = X[np.newaxis, ...]
    N, _, T = X.shape
    K = W.shape[1]
    Y = np.zeros((N, K, T))
    for n in range(N):
        Y[n, ...] = W.T.dot(X[n, ...])
    if flag:
        Y = np.squeeze(Y, axis=0)
    return Y

def get_ISC(X, W):
    '''
    Get ISC values from signal (X) and backward model (W)

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal to calculate CorrCA.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    ISC : list of floats
        Inter-subject Correlation values.
    '''
    N, D, T = X.shape

    Rw = sum(np.cov(X[n,...]) for n in range(N))
    Rt = N**2 * np.cov(X.mean(axis=0))
    Rb = (Rt - Rw) / (N-1)

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))
    return np.real(ISC)

def get_forwardmodel(X, W):
    '''
    Get forward model from signal(X) and backward model (W).

    Parameters
    ----------
    X : ndarray of shape = (n_subj, n_dim, n_times)
        Signal  to transform.
    W : ndarray of shape = (n_times, n_components)
        Backward model (signal to components).

    Returns
    -------
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).
    '''

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    Rw = sum(np.cov(X[n,...]) for n in range(N))
    Rt = N**2 * np.cov(X.mean(axis=0))
    Rb = (Rt - Rw) / (N-1)

    k = np.linalg.matrix_rank(Rw)
    if k==D:
        A = Rw.dot(W).dot(sp_linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))
    return A

def reconstruct(Y, A):
    '''
    Reconstruct signal(X) from components (Y) and forward model (A).

    Parameters
    ----------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).

    Returns
    -------
    X : ndarray of shape = (n_subj, n_dim, n_times) or (n_dim, n_times)
        Signal.
    '''

    flag = False
    if Y.ndim == 2:
        flag = True
        Y = Y[np.newaxis, ...]
    N, _, T = Y.shape
    D = A.shape[0]
    X = np.zeros((N, D, T))
    for n in range(N):
        X[n, ...] = A.dot(Y[n, ...])

    if flag:
        X = np.squeeze(X, axis=0)
    return X

def stats(X, gamma=0, k=None, n_surrogates=200, alpha=0.05):
    '''
    Compute ISC statistical threshold using circular shift surrogates.
    Parameters
    ----------
    Y : ndarray of shape = (n_subj, n_components, n_times) or (n_components, n_times)
        CorrCA components.
    A : ndarray of shape = (n_times, n_components)
        Forward model (components to signal).

    Returns
    -------
    '''
    ISC_null = []
    for n in range(n_surrogates):
        if n%10==0:
            print('#', end='')
        surrogate = circular_shift(X)
        W, ISC, A = fit(surrogate, gamma=gamma, k=k)
        ISC_null.append(ISC[0]) # get max ISC
    ISC_null = np.array(ISC_null)
    thr = np.percentile(ISC_null, (1 - alpha) * 100)
    print('')
    return thr, ISC_null

def circular_shift(X):
    n_reps, n_dims, n_times = X.shape
    shifts = np.random.choice(range(n_times), n_reps, replace=True)
    surrogate = np.zeros_like(X)
    for i in range(n_reps):
        surrogate[i, ...] = np.roll(X[i, ...], shifts[i], axis=1)
    return surrogate

def time2ix(times, t):
    return np.abs(times - t).argmin()

def get_id(params):
    CCA_id = 'CorrCA_{}_{}'.format(params['response_window'][0], params['response_window'][1])
    if params['stats']:
        CCA_id += '_stats_K_{}_surr_{}_alpha_{}_gamma_{}'.format(params['K'], params['n_surrogates'], params['alpha'], params['gamma'])
    return CCA_id

############
# PLOTTING #
############
def plot_CCA(CCA, plot_trials=True, plot_evk=False, plot_signal=False, collapse=False, xlim=(-0.3,0.6), ylim=(-7,5), norm=True, trials_alpha=0.5, width=10):
    times = CCA['times']
    
    Y = transform(CCA['epochs'], CCA['W'] )
    Ymean = np.mean(Y, axis=0)

    ISC, A, times, info = CCA['ISC'], CCA['A'], CCA['times'], CCA['info']
    n_CC = Y.shape[-2]
    print(Y.shape)

    n_rows = 2 if plot_signal else 1
    height = 6 if plot_signal else 0
    n_rows = 2 if plot_signal else 0
    height += 12 if collapse else 2.5 * n_CC
    n_rows += 2 if collapse else n_CC
    n_cols = n_CC if collapse else 3
    
    fig = plt.figure(figsize=(width, height))

    if plot_signal:
        # plot_evoked(CCA['evoked'], CCA['times'], CCA['info'], fig=fig, xlim=xlim, ylim=ylim, norm=norm)
        pass

    if CCA['W'].shape[1]!=0:
        if collapse:
            gs = fig.add_gridspec(3, min(8, n_CC), top=0.49, hspace=0.5)
            ax = fig.add_subplot(gs[:2, :])
            if plot_evk:
                ax.plot(times, CCA['evoked'].T, color='tab:grey', linewidth=0.3)

            for n in range(n_CC):
                ax.plot(times, Ymean[n, :], label = 'Component {} - ISC = {:.2f}'.format(n+1, CCA['ISC'][n]), linewidth=1.8)

            ax.legend(loc='lower left')
            ax.set_xlim(xlim)

            for n in range(min(8, n_CC)):
                vmax = np.max(np.abs(A))
                ax2 = fig.add_subplot(gs[2, n])
                im, cn = mne.viz.plot_topomap(A[:, n], pos=info, axes=ax2, show=False, vmax=vmax, vmin=-vmax)
                ax2.set_title('Component {}'.format(n+1))

                if n == n_CC-1:
                    plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.04)
        else:
            top = 0.49 if plot_signal else 0.88
            gs = fig.add_gridspec(n_CC, 3, top=top, hspace=0.3)
            for i in range(n_CC):
                ax = fig.add_subplot(gs[i, :2])

                if plot_trials:
                    trial_colors = plt.cm.viridis(np.linspace(0,1,Y.shape[0]))
                    for j in range(Y.shape[0]):
                        ax.plot(times, Y[j, i, :], linewidth=0.5, color=trial_colors[j], alpha=trials_alpha)

                if plot_evk:
                    ax.plot(times, CCA['evoked'].T, color='tab:grey', linewidth=0.3)

                ax.plot(times, Ymean[i], color='black')

                ax.set_xlim(xlim)
                ax.set_title('Component {} - ISC = {:.2f}'.format(i+1, ISC[i]))

                ax2 = fig.add_subplot(gs[i, 2])
                im, cn = mne.viz.plot_topomap(A[:, i], pos=info, axes=ax2, show=False)

    return fig



# Translation of original matlab function by Parra
def CorrCA_matlab(X, W=None, version=2, gamma=0, k=None):
    '''
    Correlated Component Analysis.

    Parameters
    ----------
    X : array, shape (n_subj, n_dim, n_times)
    k : int,
        Truncates eigenvalues on the Kth component.

    Returns
    -------
    W
    ISC
    Y
    A
    '''

    # TODO: implement case 3, tsvd truncation

    N, D, T = X.shape # subj x dim x times (instead of times x dim x subj)

    if k is not None: # truncate eigenvalues using SVD
        gamma = 0
    else:
        k = D

    # Compute within- and between-subject covariances
    if version == 1:
        Xcat = X.reshape((N * D, T)) # T x (D + N) note: dimensions vary first, then subjects
        Rkl = np.cov(Xcat).reshape((N, D, N, D)).swapaxes(1, 2)
        Rw = Rkl[range(N), range(N), ...].sum(axis=0) # Sum within subject covariances
        Rt = Rkl.reshape(N*N, D, D).sum(axis=0)
        Rb = (Rt - Rw) / (N-1)

    elif version == 2:
        Rw = sum(np.cov(X[n,...]) for n in range(N))
        Rt = N**2 * np.cov(X.mean(axis=0))
        Rb = (Rt - Rw) / (N-1)

    elif version == 3:
        pass

    if W is None:
        k = min(k, np.linalg.matrix_rank(Rw)) # handle rank deficient data.
        if k < D:
            def regInv(R, k):
                '''PCA regularized inverse of square symmetric positive definite matrix R.'''

                U, S, Vh = np.linalg.svd(R)
                invR = U[:, :k].dot(np.diag(1 / S[:k])).dot(Vh[:k, :])
                return invR

            invR = regInv(Rw, k)
            ISC, W = sp_linalg.eig(invR.dot(Rb))
            ISC, W = ISC[:k], W[:, :k]

        else:
            Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
            ISC, W = sp_linalg.eig(Rb, Rw_reg) # W is already sorted by eigenvalue and normalized

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))

    ISC, W = np.real(ISC), np.real(W)

    Y = np.zeros((N, k, T))
    for n in range(N):
        Y[n, ...] = W.T.dot(X[n, ...])

    if k==D:
        A = Rw.dot(W).dot(sp_linalg.inv(W.T.dot(Rw).dot(W)))
    else:
        A = Rw.dot(W).dot(np.diag(1 / np.diag(W.T.dot(Rw).dot(W))))

    return W, ISC, Y, A

# %%
# perform isc analysis
# convert raw data to ndarray of shape (n_subj, n_dim, n_times)
isc_results_by_event = {}
for movie_event in byd_movie_events.keys():
    print(f'Performing ISC analysis for event: {movie_event}')
    
    all_event_epoch_subjects = np.array([epochs_dict[subject_id][movie_event].get_data(picks='eeg') for subject_id in subject_ids]).squeeze()

    X = np.array(all_event_epoch_subjects)
    print('Data shape for ISC:', X.shape)  # Should be (n_subj, n_dim, n_times  )
    W, ISC, A = fit(X, k = 12)

    print(f'ISC analysis completed for event: {movie_event}')
    # plot all components' topographies
    fig = plt.figure(figsize=(12, 6))
    for i in range(W.shape[1]):
        ax = fig.add_subplot(3, 4, i+1)
        im, cn = mne.viz.plot_topomap(A[:, i], pos=epochs_dict[subject_ids[0]][movie_event].info, axes=ax, show=False, contours=8, cmap=parula_map)
        # add scale
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Component {}'.format(i+1))
    plt.show()
    isc_results_by_event[movie_event] = {
        'W': W,
        'ISC': ISC,
        'A': A
    }

