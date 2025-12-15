import streamlit as st
import numpy as np
import scipy as sp  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from scipy.linalg import eigh  # type: ignore
import networkx as nx  # type: ignore

# from timeit import default_timer
# for interactive plots, but it doesn't work great.
# import mpld3
# import streamlit.components.v1 as components
# sns.set_palette('coolwarm')
plt.style.use("dark_background")

st.title("ISC Playground")
st.link_button("Source code", "https://github.com/zeyus/isc-playground", icon="💻")
st.subheader("Updated 2025-02-10 21:41")
st.divider()

config_defaults = {
    "n_subj": 5,
    "n_chan": 2,
    "sample_rate": 5,
    "duration": "10",  # string, because it's a text input
    "n_correlated": 2,
    "signal_type": "random",
    "signal_freq": 1,
    "signal_amp": 0.5,
    "signal_phase": 0.0,
    "signal_noise": 0.01,
    "correlation": 0.5,
    "conditions": ["all"],
    "correlated_channel_groups": {},
    "W": None,
    "ISC_overall": None,
    "isc_results": None,
    "sim_data": [],
    "sim_time": None,
}
signal_types = ["sine", "square", "sawtooth", "triangle", "random"]

for key, value in config_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


with st.sidebar:
    # check if all values are defaults, if not, offer a reset button
    reset = False
    for key, value in config_defaults.items():
        if isinstance(st.session_state[key], np.ndarray):
            continue
        elif st.session_state[key] != value:
            reset = True
            break
    if reset:
        if st.button("⚠️ Reset to default settings"):
            # delete all keys and reset to defaults
            for key in st.session_state.keys():  # type: ignore
                del st.session_state[key]
            for key, value in config_defaults.items():
                st.session_state[key] = value
    with st.expander("💻 Simulation parameters", expanded=True):

        def remove_subject_from_groups():
            for cond in st.session_state.conditions:
                for i in range(st.session_state.n_subj, 10):
                    if f"subj-{i}-{cond}" in st.session_state:
                        del st.session_state[f"subj-{i}-{cond}"]
                    else:
                        break

        st.session_state.n_subj = st.select_slider(
            "Number of subjects",
            options=range(1, 11),
            value=st.session_state.n_subj,
            on_change=remove_subject_from_groups,
        )
        st.session_state.n_chan = st.select_slider(
            "Number of data channels",
            options=range(1, 65),
            value=st.session_state.n_chan,
        )
        st.session_state.sample_rate = st.select_slider(
            "Sample rate (Hz)",
            options=[5, 10, 50, 100, 250, 500, 1000],
            value=st.session_state.sample_rate,
        )
        st.session_state.duration = st.text_input(
            "Duration (seconds)", st.session_state.duration
        )

    with st.expander("📡 Signal parameters", expanded=True):
        # signal parameters
        st.session_state.signal_type = st.selectbox(
            "Signal type",
            signal_types,
            index=signal_types.index(st.session_state.signal_type),
        )
        if st.session_state.signal_type != "random":
            st.session_state.signal_freq = st.slider(
                "Signal frequency",
                min_value=1,
                max_value=st.session_state.sample_rate // 2,
                value=st.session_state.signal_freq,
            )
            st.session_state.signal_amp = st.slider(
                "Signal amplitude",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.signal_amp,
            )
            st.session_state.signal_phase = st.slider(
                "Signal phase",
                min_value=0.0,
                max_value=2.0 * np.pi,
                value=st.session_state.signal_phase,
            )
            st.session_state.signal_noise = st.slider(
                "Signal noise", min_value=0.0, max_value=0.9, value=0.01
            )

    with st.expander("🌀 Correlation parameters", expanded=True):
        # and of the signal, how many are correlated?
        corr_disabled = False
        if st.session_state.n_subj < 2:
            corr_disabled = True
        st.session_state.n_correlated = st.select_slider(
            "Number of correlated subjects",
            options=range(0, max(3, st.session_state.n_subj + 1)),
            value=st.session_state.n_correlated,
            disabled=corr_disabled,
        )

        # and of the correlated subjects, how many "spatial" groups?
        # this should somewhat translate into components in the ISC
        # these spatial groups are represented by the same channel index
        # across the correlated subjects

        # first a helper function to add or remove a group
        def add_group(group: str):
            st.session_state.correlated_channel_groups[group] = []

        def remove_group(group: str):
            del st.session_state.correlated_channel_groups[group]

        # then the actual UI
        st.subheader("Correlated channel groups")
        # add button
        col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
        new_group = ""
        with col1:
            new_group = st.text_input("New group name", key="new_group")
        with col2:
            st.button(
                "",
                key="add-group",
                icon=":material/add:",
                type="primary",
                on_click=add_group,
                args=(new_group,),
            )
        st.divider()
        for group in st.session_state.correlated_channel_groups:
            col1, col2 = st.columns([4, 1], vertical_alignment="center")
            with col1:
                st.write(f"{group}")
            with col2:
                st.button(
                    "",
                    key=f"del-{group}",
                    icon=":material/delete:",
                    type="primary",
                    on_click=remove_group,
                    args=(group,),
                )
            st.session_state.correlated_channel_groups[group] = st.multiselect(
                "Channels",
                key=f"channels-{group}",
                options=range(1, st.session_state.n_chan + 1),
                default=st.session_state.correlated_channel_groups[group],
            )
            st.divider()

        # correlation parameters
        st.session_state.correlation = st.slider(
            "Correlation",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.correlation,
        )

    with st.expander("📊 Data Parameters", expanded=True):
        st.subheader("Conditions")

        def add_cond(cond: str):
            st.session_state.conditions.append(cond)

        def remove_cond(cond: str):
            st.session_state.conditions.remove(cond)
            for i in range(st.session_state.n_subj):
                if f"subj-{i}-{cond}" in st.session_state:
                    del st.session_state[f"subj-{i}-{cond}"]
                else:
                    break

        for cond in st.session_state.conditions:
            col1, col2 = st.columns([4, 1], vertical_alignment="center")
            with col1:
                st.write(f"{cond}")
            with col2:
                st.button(
                    "",
                    key=f"del-{cond}",
                    icon=":material/delete:",
                    type="primary",
                    disabled=cond == "all",
                    on_click=remove_cond,
                    args=(cond,),
                )
            st.divider()
        col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
        new_condition = ""
        with col1:
            new_condition = st.text_input("New condition name", key="new_condition")
        with col2:
            st.button(
                "",
                key="add-cond",
                icon=":material/add:",
                type="primary",
                on_click=add_cond,
                args=(new_condition,),
            )

        tabs = st.tabs(st.session_state.conditions)
        for t, cond in enumerate(st.session_state.conditions):
            with tabs[t]:
                for i in range(st.session_state.n_subj):
                    if cond == "all":
                        st.checkbox(
                            f"subject {i}",
                            key=f"subj-{i}-{cond}",
                            value=True,
                            disabled=True,
                        )
                    else:
                        checked = (
                            f"subj-{i}-{cond}" in st.session_state
                            and st.session_state[f"subj-{i}-{cond}"]
                        )
                        st.checkbox(
                            f"subject {i}", key=f"subj-{i}-{cond}", value=checked
                        )


# generate data
@st.cache_resource
def generate_data(
    duration,
    n_subj,
    n_chan,
    sample_rate,
    n_correlated,
    signal_type,
    signal_freq,
    signal_amp,
    signal_phase,
    signal_noise,
    correlation,
    correlated_channel_groups,
):
    time = np.arange(0, int(duration) * sample_rate, 1)
    data = np.zeros((n_subj, n_chan, len(time)))
    correlated_channel_groups = correlated_channel_groups.copy()
    n_random = n_subj - n_correlated

    # generate random data
    for i in range(n_random):
        data[i] = np.random.rand(n_chan, len(time))

    # next generated the correlated subject data
    # starting with a base signal using the desired signal type
    base_signal = np.zeros((n_chan, len(time)))
    if signal_type == "sine":
        base_signal += signal_amp * np.sin(
            2 * np.pi * signal_freq * time / sample_rate + signal_phase
        )
    elif signal_type == "square":
        base_signal += signal_amp * sp.signal.square(
            2 * np.pi * signal_freq * time / sample_rate + signal_phase
        )
    elif signal_type == "sawtooth":
        base_signal += signal_amp * sp.signal.sawtooth(
            2 * np.pi * signal_freq * time / sample_rate + signal_phase
        )
    elif signal_type == "triangle":
        base_signal += signal_amp * sp.signal.sawtooth(
            2 * np.pi * signal_freq * time / sample_rate + signal_phase,
            width=0.5,
        )
    elif signal_type == "random":
        base_signal += np.random.rand(n_chan, len(time)) * signal_amp

    # add noise to the base signal
    if signal_type != "random" and signal_noise > 0.0:
        base_signal += np.random.rand(n_chan, len(time)) * signal_noise

    if n_correlated > 1:
        # generate correlated data
        data[n_random] = base_signal
        group_bases = dict()
        chan_level = False
        remove_groups = []
        if len(correlated_channel_groups.keys()) > 0:
            for group, channels in correlated_channel_groups.items():
                if len(channels) > 1:
                    chan_level = True
                else:
                    # remove the group
                    remove_groups.append(group)
                    st.warning(
                        f"Group {group} has only less than two channels. It will be ignored.")
            for group in remove_groups:
                del correlated_channel_groups[group]

        if chan_level:
            for group, channels in correlated_channel_groups.items():
                # turn string channels into integers, and subtract 1 to get 0-based index
                channels = [int(c) - 1 for c in channels]
                group_base = np.random.rand(len(time)) * signal_amp
                # create the correlated signal between the channels
                group_bases[group] = np.zeros((len(channels), len(time)))
                for i, _ in enumerate(channels):
                    group_bases[group][i, :] = group_base * correlation + np.random.rand(
                        len(time)
                    ) * (1 - correlation)

        for i in range(n_random + 1, n_subj):
            # now create remaining correlated data for each subject and the channel groups (if any)
            if chan_level:
                data[i] = np.random.rand(n_chan, len(time)) * signal_amp
                for group, channels in correlated_channel_groups.items():
                    # turn string channels into integers, and subtract 1 to get 0-based index
                    channels = [int(c) - 1 for c in channels]
                    st.write(f"assigning group {group} to channels {channels} for subject {i}")
                    data[i][channels, :] = group_bases[group] * correlation + np.random.rand(
                        len(channels), len(time)
                    ) * (1 - correlation)
            else:
                data[i] = base_signal * correlation + np.random.rand(n_chan, len(time)) * (
                    1 - correlation
                )

    return data, time


# from: https://github.com/ML-D00M/ISC-Inter-Subject-Correlations/blob/main/Python/ISC.py
@st.cache_resource
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


@st.cache_resource
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
    A = np.linalg.solve(Rw @ W, np.transpose(W) @ Rw @ W)

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


@st.cache_resource
def plot_data(data, time, subject_selection, channel_selection):
    n_sel = len(subject_selection)
    fig, ax = plt.subplots(n_sel, figsize=(10, 10))

    for i in range(n_sel):
        # plot all channels for each subject
        axi = ax[i] if n_sel > 1 else ax
        for j in range(len(channel_selection)):
            axi.plot(time, data[subject_selection[i], channel_selection[j], :])
        axi.set_ylabel(f"S{subject_selection[i] + 1}")
    fig.supxlabel("Time (ms)")
    fig.supylabel("Amplitude")

    plt.tight_layout()
    return fig


@st.cache_resource
def plot_corrmatrix(data, subject_selection):
    n_sel = len(subject_selection)
    corr = np.zeros((n_sel, n_sel))
    for i in range(n_sel):
        for j in range(n_sel):
            corr[i, j] = np.corrcoef(
                data[subject_selection[i]].flatten(),
                data[subject_selection[j]].flatten(),
            )[0, 1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(
        corr,
        vmin=0.0,
        vmax=1.0,
        cmap="coolwarm",
        ax=ax,
        yticklabels=[f"S{i + 1}" for i in subject_selection],
        xticklabels=[f"S{i + 1}" for i in subject_selection],
    )

    ax.set_title("Subject Correlation Matrix")
    return fig, corr


@st.cache_resource
def plot_chann_corrmatrix(data, channel_selection):
    n_sel = len(channel_selection)
    corr = np.zeros((n_sel, n_sel))
    for i in range(n_sel):
        for j in range(n_sel):
            corr[i, j] = np.corrcoef(
                data[:, channel_selection[i]].flatten(),
                data[:, channel_selection[j]].flatten(),
            )[0, 1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(
        corr,
        vmin=0.0,
        vmax=1.0,
        cmap="coolwarm",
        ax=ax,
        yticklabels=[f"Ch{i + 1}" for i in channel_selection],
        xticklabels=[f"Ch{i + 1}" for i in channel_selection],
    )

    ax.set_title("Channel Correlation Matrix")
    return fig, corr


@st.cache_resource
def plot_network_from_corr(corr, remove_self=True, shell_layout=True):
    # generate a network graph from the correlation matrix
    G = nx.from_numpy_array(corr)

    # add weight to edges and set line width based on weight
    for i, j in G.edges:
        G[i][j]["weight"] = corr[i, j]

    if remove_self:
        G.remove_edges_from(nx.selfloop_edges(G))

    widths = nx.get_edge_attributes(G, "weight")
    nodelist = G.nodes()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if shell_layout:
        pos = nx.shell_layout(G, scale=10)
    else:
        pos = nx.spring_layout(G, scale=10)

    nx.draw_networkx_nodes(
        G, pos, nodelist=nodelist, node_color="#440022", node_size=500, alpha=0.8, ax=ax
    )
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.4,
        edge_vmin=0.0,
        edge_vmax=1.0,
        edge_cmap=plt.cm.coolwarm,
        edge_color=list(widths.values()),
        width=[v * 15 for v in widths.values()],
        edgelist=widths.keys(),
        ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_color="white", labels={i: f"{i + 1}" for i in nodelist}
    )

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, alpha=0.4)
    ax.set_title("Network from Correlation Matrix")

    return fig


@st.cache_resource
def plot_power(data, subject_selection, channel_selection):
    fig, ax = plt.subplots(n_sel, figsize=(10, 10))
    for i in range(n_sel):
        axi = ax[i] if n_sel > 1 else ax
        # plot all channels for each subject
        for j in range(len(channel_selection)):
            f, Pxx = sp.signal.welch(
                data[subject_selection[i], channel_selection[j], :],
                fs=st.session_state.sample_rate,
                nperseg=st.session_state.sample_rate,
            )
            axi.plot(f, Pxx)
        axi.set_ylabel(f"S{subject_selection[i] + 1}")
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Power")
    return fig


def format_subj(subj):
    return f"S{subj + 1}"


def format_chan(chan):
    return f"Ch{chan + 1}"

data = st.session_state.sim_data
time = st.session_state.sim_time
st.subheader("Generated Data")

button_label = "Generate data" if data is None else "Re-generate data"
if len(data) == 0:
    st.write("No data generated yet, click the button below to generate data.")
if st.button(button_label, type="primary", key="generate_data"):
    # generate data based on the current settings
    data, time = generate_data(
        st.session_state.duration,
        st.session_state.n_subj,
        st.session_state.n_chan,
        st.session_state.sample_rate,
        st.session_state.n_correlated,
        st.session_state.signal_type,
        st.session_state.signal_freq,
        st.session_state.signal_amp,
        st.session_state.signal_phase,
        st.session_state.signal_noise,
        st.session_state.correlation,
        st.session_state.correlated_channel_groups,
    )
    st.session_state.sim_data = data
    st.session_state.sim_time = time

if len(data) > 0 and time is not None:
    data_tab, sub_tab, chan_tab, power_tab = st.tabs(
        ["Data", "Subject Correlation", "Channel Correlation", "Power spectrum"]
    )

    # exploration of generated data
    with data_tab:
        subject_selection = st.pills(
            "Subjects",
            range(st.session_state.n_subj),
            format_func=format_subj,
            default=range(st.session_state.n_subj),
            selection_mode="multi",
        )
        channel_selection = st.pills(
            "Channels",
            range(st.session_state.n_chan),
            format_func=format_chan,
            default=range(st.session_state.n_chan),
            selection_mode="multi",
        )

        # ifig = mpld3.fig_to_html(fig)
        # components.html(ifig, height=600)
        fig = plot_data(data, time, subject_selection, channel_selection)
        st.pyplot(fig)

    with sub_tab:
        # plot correlation matrix
        subject_selection = st.pills(
            "Subjects",
            range(st.session_state.n_subj),
            key="subjects_corplot",
            format_func=format_subj,
            default=range(st.session_state.n_subj),
            selection_mode="multi",
        )

        fig, corr = plot_corrmatrix(data, subject_selection)
        st.pyplot(fig)

        spring_layout = st.toggle("Spring layout", key="spring_subj", value=False)
        fig = plot_network_from_corr(corr, shell_layout=not spring_layout)
        st.pyplot(fig)

    with chan_tab:
        channel_selection = st.pills(
            "Channels",
            range(st.session_state.n_chan),
            key="channels_corplot",
            format_func=format_chan,
            default=range(st.session_state.n_chan),
            selection_mode="multi",
        )
        fig, corr = plot_chann_corrmatrix(data, channel_selection)
        st.pyplot(fig)

        spring_layout = st.toggle("Spring layout", key="spring_chan", value=False)
        fig = plot_network_from_corr(corr, shell_layout=not spring_layout)
        st.pyplot(fig)

        # ifig2 = mpld3.fig_to_html(fig)
        # components.html(ifig2, height=600)

    with power_tab:
        # plot power spectrum
        subject_selection = st.pills(
            "Subjects",
            range(st.session_state.n_subj),
            key="subjects_powerplot",
            format_func=format_subj,
            default=range(st.session_state.n_subj),
            selection_mode="multi",
        )
        n_sel = len(subject_selection)
        channel_selection = st.pills(
            "Channels",
            range(st.session_state.n_chan),
            key="channels_powerplot",
            format_func=format_chan,
            default=range(st.session_state.n_chan),
            selection_mode="multi",
        )

        fig = plot_power(data, subject_selection, channel_selection)
        st.pyplot(fig)


# helper to get grouped subjects
def get_subjs_by_cond(conditions):
    subj_by_cond = dict()
    for cond in conditions:
        subj_by_cond[cond] = [
            i
            for i in range(st.session_state.n_subj)
            if st.session_state[f"subj-{i}-{cond}"]
        ]
    return subj_by_cond


# prepare conditions
@st.cache_resource
def prepare_conditions(data, conditions, included_subj):
    data_dict = dict()
    for cond in conditions:
        if cond == "all":
            data_dict[cond] = data
            continue
        if cond in included_subj and len(included_subj[cond]) > 2:
            data_dict[cond] = data[included_subj[cond]]
    return data_dict


@st.cache_resource
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


# plot ISC over time
@st.cache_resource
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

@st.cache_resource
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


# plot spatial filter weights
@st.cache_resource
def plot_weights(W):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(W, ax=ax)
    ax.set_title("Spatial filter weights")
    # add x and y labels
    ax.set_xlabel("Component")
    ax.set_ylabel("Channel")
    
    return fig


if data is not None:
    included_subjects = get_subjs_by_cond(st.session_state.conditions)
    data_dict = prepare_conditions(data, st.session_state.conditions, included_subjects)
    st.subheader("Inter-Subject Correlation Analysis")
    isc_ready = st.session_state.W is not None and st.session_state.ISC_overall is not None
    run_label = "Run ISC" if not isc_ready else "Re-run ISC"
    if st.button(run_label, key="run_isc", type="primary"):
        # get the spatial filter weights and ISC values
        [st.session_state.W, st.session_state.ISC_overall] = train_cca(data_dict)
        st.session_state.isc_results = dict()

        # apply the spatial filter weights to the data by condition
        for cond_key, cond_values in data_dict.items():
            st.session_state.isc_results[str(cond_key)] = dict(
                zip(
                    ["ISC", "ISC_persecond", "ISC_bysubject", "A"],
                    apply_cca(
                        cond_values, st.session_state.W, st.session_state.sample_rate
                    ),
                )
            )
        st.write("ISC completed")

    # display the results
    if st.session_state.W is not None and st.session_state.ISC_overall is not None:
        # show the ISC values as a table for each component
        df = {f"C{i + 1}": [v] for i, v in enumerate(st.session_state.ISC_overall)}
        st.dataframe(df)
        filter_weight_tab, isc_summary_tab, isc_time_tab, subject_tab = st.tabs(
            ["Filter weights", "ISC summary", "ISC over time", "ISC by subject"]
        )
        with filter_weight_tab:
            # plot spatial filter weights
            fig = plot_weights(st.session_state.W)
            st.pyplot(fig)

            w_component = st.select_slider(
                "Component/Channel",
                options=range(1, len(st.session_state.ISC_overall) + 1),
                value=1,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"Component {w_component} weights")
                df = {
                    f"Ch{i + 1}": v
                    for i, v in enumerate(st.session_state.W[:, w_component - 1])
                }
                st.dataframe(df)

            with c2:
                st.write(f"Channel {w_component} weights")
                df = {
                    f"Comp{i + 1}": v
                    for i, v in enumerate(st.session_state.W[w_component - 1, :])
                }
                st.dataframe(df)

        with isc_summary_tab:
            # plot ISC summary
            fig = plot_isc(st.session_state.isc_results)
            st.pyplot(fig)

        with isc_time_tab:
            # plot ISC over time
            fig = plot_isc_time(st.session_state.isc_results)
            st.pyplot(fig)
        
        with subject_tab:
            # plot ISC by subject
            fig = plot_isc_subj(st.session_state.isc_results)
            st.pyplot(fig)
