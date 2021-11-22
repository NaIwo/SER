SEARCH_FOLDER="../datasets/datasets_files/MELD/raw_data"

recursive() {
  for d in *; do
    if [ -d "$d" ]; then
      (cd -- "$d" && recursive)
    fi
    if [[ $d =~ \.mp4$ ]]; then
      filename="$(basename $d .mp4)"
      ffmpeg -i "$d" -ac 1 -f wav "$filename".wav
      rm "$d"
    fi
  done
}

(cd $SEARCH_FOLDER; recursive)