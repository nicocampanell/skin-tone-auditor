"""
Fitzpatrick17k Image Downloader
================================
Downloads all images from URLs in fitzpatrick17k.xlsx,
saving each file as <md5hash>.jpg in the output directory.

Usage
-----
    python download_images.py

    # Custom paths:
    python download_images.py --xlsx fitzpatrick17k.xlsx --output images/

    # Resume interrupted download (skips already downloaded):
    python download_images.py --resume

    # Limit workers (default 8, reduce if getting rate-limited):
    python download_images.py --workers 4

    # Only download QC-passed images:
    python download_images.py --qc-only
"""

import os
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
DEFAULT_XLSX   = "fitzpatrick17k.xlsx"
DEFAULT_OUTDIR = "images/"
MAX_WORKERS    = 8       # concurrent download threads
TIMEOUT        = 15      # seconds per request
MAX_RETRIES    = 3       # retry failed downloads
RETRY_DELAY    = 2       # seconds between retries

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOADER
# ══════════════════════════════════════════════════════════════════════════════

def download_image(row: dict, output_dir: str, resume: bool = True) -> dict:
    """
    Download a single image. Returns a result dict with status.

    Parameters
    ----------
    row        : dict with keys 'md5hash' and 'url'
    output_dir : directory to save images
    resume     : skip if file already exists on disk
    """
    md5  = str(row['md5hash'])
    url  = str(row['url'])
    dest = os.path.join(output_dir, md5 + '.jpg')

    # Skip if already downloaded
    if resume and os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return {'md5': md5, 'status': 'skipped', 'url': url}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()

            # Validate content type is an image
            content_type = resp.headers.get('Content-Type', '')
            if 'image' not in content_type and attempt == MAX_RETRIES:
                return {'md5': md5, 'status': 'not_image',
                        'url': url, 'error': f'Content-Type: {content_type}'}

            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Sanity check: file should be > 1KB
            if os.path.getsize(dest) < 1000:
                os.remove(dest)
                return {'md5': md5, 'status': 'too_small', 'url': url}

            return {'md5': md5, 'status': 'success', 'url': url}

        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES:
                return {'md5': md5, 'status': 'timeout', 'url': url, 'error': 'Timeout'}
            time.sleep(RETRY_DELAY * attempt)

        except requests.exceptions.HTTPError as e:
            # Don't retry 404s
            if e.response.status_code == 404:
                return {'md5': md5, 'status': '404', 'url': url, 'error': str(e)}
            if attempt == MAX_RETRIES:
                return {'md5': md5, 'status': 'http_error', 'url': url, 'error': str(e)}
            time.sleep(RETRY_DELAY * attempt)

        except Exception as e:
            if attempt == MAX_RETRIES:
                return {'md5': md5, 'status': 'failed', 'url': url, 'error': str(e)}
            time.sleep(RETRY_DELAY * attempt)

    return {'md5': md5, 'status': 'failed', 'url': url}


def download_all(xlsx_path: str, output_dir: str,
                 resume: bool = True, workers: int = MAX_WORKERS,
                 qc_only: bool = False) -> pd.DataFrame:
    """
    Download all images from the spreadsheet in parallel.

    Returns a DataFrame with per-image download status.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load spreadsheet
    print(f"\n[Loading] {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    print(f"[Loaded]  {len(df):,} rows | Columns: {df.columns.tolist()}")

    # Filter invalid URLs
    df = df[df['url'].notna() & (df['url'].astype(str).str.startswith('http'))]
    print(f"[Valid URLs] {len(df):,} rows after filtering nulls")

    # Optional: QC filter
    if qc_only and 'qc' in df.columns:
        df = df[df['qc'] == 'pass']
        print(f"[QC Filter] {len(df):,} rows after QC=pass filter")

    rows = df[['md5hash', 'url']].to_dict('records')

    # Check how many are already downloaded
    already = sum(
        1 for r in rows
        if os.path.exists(os.path.join(output_dir, str(r['md5hash']) + '.jpg'))
        and os.path.getsize(os.path.join(output_dir, str(r['md5hash']) + '.jpg')) > 1000
    )
    to_download = len(rows) - already if resume else len(rows)
    print(f"\n[Status] Already downloaded : {already:,}")
    print(f"[Status] To download        : {to_download:,}")
    print(f"[Status] Workers            : {workers}")
    print(f"[Status] Output dir         : {output_dir}\n")

    # Download in parallel
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image, row, output_dir, resume): row
            for row in rows
        }
        with tqdm(total=len(rows), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                if result['status'] not in ('success', 'skipped'):
                    pbar.set_postfix({'last_fail': result['status']})

    # ── Summary ────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    counts = results_df['status'].value_counts()

    print(f"\n{'='*50}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*50}")
    print(f"  Success   : {counts.get('success', 0):>6,}")
    print(f"  Skipped   : {counts.get('skipped', 0):>6,}  (already existed)")
    print(f"  404       : {counts.get('404', 0):>6,}")
    print(f"  Timeout   : {counts.get('timeout', 0):>6,}")
    print(f"  Failed    : {counts.get('failed', 0):>6,}")
    print(f"  Not image : {counts.get('not_image', 0):>6,}")
    print(f"  Too small : {counts.get('too_small', 0):>6,}")
    print(f"{'='*50}")

    total_on_disk = sum(
        1 for r in rows
        if os.path.exists(os.path.join(output_dir, str(r['md5hash']) + '.jpg'))
    )
    print(f"\n  Total images on disk: {total_on_disk:,} / {len(rows):,}")
    print(f"  Output directory    : {os.path.abspath(output_dir)}")

    # Save log
    log_path = os.path.join(output_dir, 'download_log.csv')
    results_df.to_csv(log_path, index=False)
    print(f"  Download log saved  : {log_path}")

    # Save failed URLs for easy retry
    failed = results_df[~results_df['status'].isin(['success', 'skipped'])]
    if len(failed) > 0:
        fail_path = os.path.join(output_dir, 'failed_downloads.csv')
        failed.to_csv(fail_path, index=False)
        print(f"  Failed URLs saved   : {fail_path}  ({len(failed):,} images)")
        print(f"\n  To retry failed: python download_images.py --resume")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Fitzpatrick17k Image Downloader")
    parser.add_argument('--xlsx',    type=str, default=DEFAULT_XLSX,
                        help=f'Path to spreadsheet (default: {DEFAULT_XLSX})')
    parser.add_argument('--output',  type=str, default=DEFAULT_OUTDIR,
                        help=f'Image output directory (default: {DEFAULT_OUTDIR})')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Parallel download threads (default: {MAX_WORKERS})')
    parser.add_argument('--resume',  action='store_true', default=True,
                        help='Skip already-downloaded images (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Re-download everything from scratch')
    parser.add_argument('--qc-only', action='store_true',
                        help='Only download images where qc=pass')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    resume = not args.no_resume

    download_all(
        xlsx_path  = args.xlsx,
        output_dir = args.output,
        resume     = resume,
        workers    = args.workers,
        qc_only    = args.qc_only,
    )
