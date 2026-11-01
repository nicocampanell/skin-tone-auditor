#!/usr/bin/env bash
# Run the models on images in gs://$bucket/images/ that aren't in results/results.csv yet, and append a row for each.
# The bucket's results.csv is the source of truth: delete a row there and that image is rerun next time.
set -euo pipefail

bucket=colorimetry-ncs325
repo=$HOME/nn_colorimetry_dermatoscopy
data=$HOME/colorimetry-data

mkdir -p "$data/images"
gcloud storage rsync --delete-unmatched-destination-objects "gs://$bucket/images" "$data/images"
gcloud storage cp "gs://$bucket/results/results.csv" "$data/results.csv" 2>/dev/null || rm -f "$data/results.csv"
"$repo/.venv/bin/python" "$repo/run_images.py" --csv "$data/results.csv" \
  --link-prefix "https://storage.cloud.google.com/$bucket/images/" "$data/images"
gcloud storage cp "$data/results.csv" "gs://$bucket/results/results.csv"
