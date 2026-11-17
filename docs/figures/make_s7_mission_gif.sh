#!/bin/bash
# Make docs/figures/s7_full_mission.gif (README) from the S7 mission video
# (runs/s7_full_mission_video_seed0_video/mission.mp4, scripts/render_mission_video.py).
# Parts of the video: descent orbit 4.2-38.4 s (sped up 8x more), powered descent 40.4-102.6 s
# (12x), 3D terminal phase 106-221 s (16x); 560 px, 10 frames/s, one 64-colour palette per frame.
set -e
VIDEO=${1:-runs/s7_full_mission_video_seed0_video/mission.mp4}
ffmpeg -v error -y -i "$VIDEO" -filter_complex "\
[0:v]trim=4.2:38.4,setpts=(PTS-STARTPTS)/8[a];\
[0:v]trim=40.4:102.6,setpts=(PTS-STARTPTS)/12[b];\
[0:v]trim=106.0:221.0,setpts=(PTS-STARTPTS)/16[c];\
[a][b][c]concat=n=3:v=1:a=0,fps=10,scale=560:-1:flags=lanczos,split[x][y];[x]palettegen=max_colors=64:stats_mode=single[p];[y][p]paletteuse=new=1:dither=bayer:bayer_scale=5" \
  -loop 0 docs/figures/s7_full_mission.gif
