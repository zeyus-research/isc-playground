#!/usr/bin/env fish

# Check for ffprobe
if not type -q ffprobe
    echo "Error: ffprobe is not installed or not in PATH."
    echo "Please install FFmpeg to use this script."
    exit 1
end

# video file as first argument
set video_file $argv[1]
set out_dir $argv[2]

if test -z $video_file
    echo "Usage: frame_analysis.fish <video_file> [out_dir]"
    echo "Extracts per-frame luminance, loudness, and amplitude features from the given video file."
    echo "Outputs CSV files to the specified out_dir or the video's directory if not provided."
    exit 1
end

if test -z $out_dir
    set out_dir (dirname $video_file)
else if test -n $out_dir
    mkdir -p $out_dir
    cd $out_dir
end

set out_base (basename $video_file | sed 's/\.[^.]*$//')

echo "Processing video file: $video_file"
echo "Output directory: $out_dir"
echo "Analysing luminance..."
ffprobe -f lavfi -i "movie=$video_file,signalstats" \
        -show_entries frame=pkt_pts_time:frame_tags=lavfi.signalstats.YAVG,lavfi.signalstats.YMIN,lavfi.signalstats.YMAX,lavfi.signalstats.YDIF \
        -of csv=p=1 > "$out_dir/$out_base-luminance.csv" 2>/dev/null
echo "Analysing loudness..."
ffprobe -f lavfi -i "amovie=$video_file,pan=mono|c0=FL,ebur128=metadata=1" \
        -show_entries frame=pkt_pts_time:frame_tags=lavfi.r128.M \
        -of csv=p=1 > "$out_dir/$out_base-loudness(EBUR128,LUFS).csv" 2>/dev/null
echo "Analysing amplitude..."
ffprobe -f lavfi -i "amovie=$video_file,pan=mono|c0=FL,astats=metadata=1:reset=1" \
        -show_entries frame=pkt_pts_time:frame_tags=lavfi.astats.Overall.RMS_level,lavfi.astats.Overall.Peak_level \
        -of csv=p=1 > "$out_dir/$out_base-amplitude.csv" 2>/dev/null

echo "Analysis complete. Output files:"
echo " - $out_dir/$out_base-luminance.csv"
echo " - $out_dir/$out_base-loudness(EBUR128,LUFS).csv"
echo " - $out_dir/$out_base-amplitude.csv"
echo "You can now run 'uv run python src/composite-stimuli-features.py' to combine and visualize these features."
