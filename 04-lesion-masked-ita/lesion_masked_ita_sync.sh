#!/usr/bin/env bash
set -euo pipefail

bucket=colorimetry-ncs325
repo="$HOME/nn_colorimetry_dermatoscopy"
data="$HOME/colorimetry-data"
images="$data/masked-images"
overlays="$data/masked-overlays"
results="$data/masked-results.csv"

mkdir -p "$images" "$overlays"
gcloud storage rsync --delete-unmatched-destination-objects "gs://$bucket/masked-images" "$images"
gcloud storage cp "gs://$bucket/masked-results/results.csv" "$results" 2>/dev/null || rm -f "$results"

"$repo/.venv/bin/python" "$repo/run_images.py" "$images" \
  --mask-ita \
  --mask-backend kmeans \
  --csv "$results" \
  --overlay-dir "$overlays" \
  --link-prefix "https://storage.cloud.google.com/$bucket/masked-images/" \
  --overlay-link-prefix "https://storage.cloud.google.com/$bucket/masked-results/overlays/"

gcloud storage rsync "$overlays" "gs://$bucket/masked-results/overlays"
gcloud storage cp "$results" "gs://$bucket/masked-results/results.csv"
