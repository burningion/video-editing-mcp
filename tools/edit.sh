# Get the highest resolution from all input files
max_width=0
max_height=0

while IFS= read -r line; do
    if [[ $line =~ ^file ]]; then
        file=$(echo "$line" | cut -d "'" -f 2)
        dims=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$file")
        width=$(echo $dims | cut -d "x" -f 1)
        height=$(echo $dims | cut -d "x" -f 2)
        
        if [ $width -gt $max_width ]; then
            max_width=$width
        fi
        if [ $height -gt $max_height ]; then
            max_height=$height
        fi
    fi
done < videos.txt

# Use the detected dimensions in the FFmpeg command
ffmpeg -f concat -safe 0 -i videos.txt \
  -vf "fps=30,scale=${max_width}:${max_height}:force_original_aspect_ratio=decrease,pad=${max_width}:${max_height}:(ow-iw)/2:(oh-ih)/2" \
  -c:v h264_videotoolbox -b:v 8M \
  -c:a aresample=async=1000 -ar 48000 \
  output.mp4