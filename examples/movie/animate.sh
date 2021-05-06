#!/bin/bash

# # Convert region overlay SVG to PNG
# convert cortical_map_top_down.svg -resize 200% cortical_map_top_down.png

# Remove whitespace from images
cd images
for im_f in $(ls *.png); do
    echo $im_f
    convert $im_f -trim -bordercolor White -border 10 -resize 1280x720 -background white -gravity center -extent 1280x720 $im_f
done

# Make video
ffmpeg -start_number 0 -i "%05d.png" -c:v libx265 -preset slow -crf 25 -r 10 ../out.avi

ffmpeg -start_number 0 -i "%05d.png" -c:v libaom-av1 -crf 30 -b:v 0 -r 10 ../out.mp4

