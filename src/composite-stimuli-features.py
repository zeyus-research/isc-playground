import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from argparse import ArgumentParser

def load_and_interpolate_to_reference(
    file_path: Path, 
    reference_timestamps: np.ndarray,
    value_columns: list[str]
) -> pd.DataFrame:
    """Load a CSV and interpolate values to reference timestamps."""
    # Columns: 'frame' literal, timestamp, then values...
    df = pd.read_csv(file_path, header=None)
    df = df.drop(columns=[0])  # drop the literal 'frame' column
    df.columns = ['timestamp'] + value_columns  # type: ignore
    
    # Interpolate each value column to reference timestamps
    result = pd.DataFrame({'timestamp': reference_timestamps})
    for col in value_columns:
        result[col] = np.interp(reference_timestamps, df['timestamp'], df[col])
    
    return result


def main(data_dir: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    
    stimuli = {
        "BangBangYouAreDead": "BangBangYouAreDead_SerialTrigInterval-1sec",
        "StoryCorps_Q&A": "StoryCorps_Q&A_SerialTrigInterval-1sec",
    }
    
    for stimulus_name, file_base in stimuli.items():
        # Load video luminance first as the reference timebase
        lum_df = pd.read_csv(data_dir / f"{file_base}-luminance.csv", header=None)
        lum_df = lum_df.drop(columns=[0])  # drop 'frame' literal
        lum_df.columns = ['timestamp', 'min_lum', 'mean_lum', 'max_lum', 'diff_lum']
        
        reference_timestamps = lum_df['timestamp'].values
        
        # Interpolate audio amplitude to video timestamps
        amp_df = load_and_interpolate_to_reference(
            data_dir / f"{file_base}-amplitude.csv",
            reference_timestamps,
            ['amp_rms', 'amp_peak']
        )
        
        # Interpolate loudness to video timestamps
        loud_df = load_and_interpolate_to_reference(
            data_dir / f"{file_base}-loudness(EBUR128,LUFS).csv",
            reference_timestamps,
            ['ebu_r128_M']
        )
        
        # Merge all on timestamp
        composite_df = lum_df.copy()
        composite_df = composite_df.merge(amp_df, on='timestamp')
        composite_df = composite_df.merge(loud_df, on='timestamp')
        
        output_file = out_dir / f"{stimulus_name}_composite_frame_level_analysis.csv"
        composite_df.to_csv(output_file, index=False)
        print(f"Saved composite frame-level analysis to {output_file}")
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        axes[0].plot(composite_df['timestamp'], composite_df['amp_rms'], label='RMS', color='blue', alpha=0.6)
        axes[0].plot(composite_df['timestamp'], composite_df['amp_peak'], label='Peak', color='green', alpha=0.5)
        # add 5s moving average for RMS
        window_size = int(5 / np.median(np.diff(composite_df['timestamp'])))
        rms_moving_avg = composite_df['amp_rms'].rolling(window=window_size, min_periods=1).mean()
        axes[0].plot(composite_df['timestamp'], rms_moving_avg, label='RMS 5s MA', color='purple', linestyle='--', linewidth=1)
        axes[0].set_ylabel('Amplitude (dBFS)')
        axes[0].set_title(f'{stimulus_name} - Audio Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(composite_df['timestamp'], composite_df['ebu_r128_M'], label='Momentary Loudness', color='purple', alpha=0.7)
        # add 5s moving average for loudness
        loud_moving_avg = composite_df['ebu_r128_M'].rolling(window=window_size, min_periods=1).mean()
        axes[1].plot(composite_df['timestamp'], loud_moving_avg, label='Momentary Loudness 5s MA', color="#00C093", linestyle='--', linewidth=1)
        axes[1].set_ylabel('Loudness (LUFS)')
        axes[1].set_title('Perceptual Loudness (EBU R128)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].fill_between(composite_df['timestamp'], composite_df['min_lum'], composite_df['max_lum'], alpha=0.3, color='orange', label='Min-Max Range')
        axes[2].plot(composite_df['timestamp'], composite_df['mean_lum'], label='Mean', color='orange', alpha=0.6)
        # add 5s moving average for mean luminance
        lum_moving_avg = composite_df['mean_lum'].rolling(window=window_size, min_periods=1).mean()
        axes[2].plot(composite_df['timestamp'], lum_moving_avg, label='Mean 5s MA', color='red', linestyle='--', linewidth=1)
        axes[2].set_ylabel('Luminance (0-255)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title('Video Luminance')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = out_dir / f"{stimulus_name}_composite_time_series.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Saved plot to {plot_file}")
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Composite Stimuli Features")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing input files")
    parser.add_argument("--out_dir", type=Path, required=False, help="Directory to save output files", default="./out")
    args = parser.parse_args()

