"""
generate_manifest_test.py
--------------------------
Test run of the manifest generator — processes 30 images only.
Outputs to test_manifest.csv / test_prompts.jsonl / test_rejected.jsonl
so it doesn't touch the full-run files.

Usage:
    python3 generate_manifest_test.py
"""

import asyncio
import base64
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage
from tqdm.asyncio import tqdm_asyncio
from google import genai
from google.genai import types as genai_types

# ── Config ─────────────────────────────────────────────────────────────────────

PROJECT_ID      = "skin-tone-classifier"
VERTEX_LOCATION = "us-central1"
IMAGE_BUCKET    = "fitz_clean"
IMAGE_PREFIX    = "images/"
CSV_PATH        = "fp17kc.csv"

OUTPUT_MANIFEST = "test_manifest.csv"
PROMPTS_CACHE   = "test_prompts.jsonl"
REJECTED_LOG    = "test_rejected.jsonl"

TARGET_FST      = ["FP5", "FP6"]
TEST_LIMIT      = 30               # number of images to process
VARIANTS_PER    = 1
BASE_SEED       = 42
MAX_CONCURRENT  = 5

SD_STRENGTH          = 0.30
SD_GUIDANCE_SCALE    = 7.5
SD_CONTROLNET_SCALE  = 0.80
SD_STEPS             = 50

MIN_DIMENSION    = 224

GEMINI_MODEL     = "gemini-2.5-flash"
CLAUDE_RESIZE_PX = 1024
THINKING_BUDGET  = 1024
WATERMARK_CROP_PCT = 0.10

# ── FST descriptions ────────────────────────────────────────────────────────────

FST_DESCRIPTIONS = {
    "FP1": "Fitzpatrick type I, very fair skin, ivory tone",
    "FP2": "Fitzpatrick type II, fair skin, light beige tone",
    "FP3": "Fitzpatrick type III, medium skin, light brown tone",
    "FP4": "Fitzpatrick type IV, olive skin, moderate brown tone",
    "FP5": "Fitzpatrick type V, brown skin, deep tan tone",
    "FP6": "Fitzpatrick type VI, deep brown skin, very dark brown tone",
}

UNIVERSAL_NEGATIVE = (
    "jewelry, metal, shiny, blurry, low resolution, distorted, "
    "unnatural colors, deformed anatomy, extra digits, writing, "
    "overlay, text, logo, SFS, watermark, ruler, gloves, plastic skin"
)

GEMINI_PROMPT = """You are an expert clinical dermatology image analyst assisting with research.

You will receive (a) a dermatology image, (b) the ground-truth disease label from
the dataset, and (c) the disease category. Return ONLY a valid JSON object with
exactly these fields:

{
  "orientation":        "<dorsal/palmar/lateral/frontal view>",
  "laterality":         "<left/right/both/unspecified>",
  "body_part":          "<specific anatomical location, e.g. dorsal forearm, lower leg>",
  "primary_lesion":     "<primary lesion morphology, e.g. papules, plaques, vesicles>",
  "shape":              "<lesion shape, e.g. annular, round, ovoid, targetoid, polycyclic>",
  "configuration":      "<arrangement of lesions, e.g. clustered, grouped, linear, discrete>",
  "lesion_color":       "<specific color, e.g. erythematous, violaceous, dusky, purpuric>",
  "texture":            "<surface/palpation quality, e.g. dome-shaped, indurated, vegetative>",
  "border_character":   "<border description, e.g. well-defined, irregular, scalloped>",
  "distribution":       "<body location pattern, e.g. acral, intertriginous, extensor>",
  "secondary_features": "<secondary changes incl. scale, e.g. crusting with fine white scale>",
  "background_skin":    "<uninvolved skin description, e.g. smooth, xerotic>",
  "sd_prompt_body":     "<flowing clinical description for Stable Diffusion, 30-45 words — see examples>"
}

How to fill each field:

1. BE CONCISE. Structured slots (orientation through background_skin) must be
   at most 5 words each. These are for the structured CSV manifest.

2. body_part must name ONE primary anatomical region (e.g. "dorsal forearm",
   "elbow", "shin"). Do not list multiple regions or fields of view.

3. VISUAL EVIDENCE FIRST. For fields clearly visible in the image, describe
   exactly what you see.

4. USE THE DISEASE LABEL AS A CLINICAL PRIOR. The disease label and category
   are ground truth from the dataset. For fields that are ambiguous, partially
   obscured, or structurally hard to judge from a single photo — especially
   texture, secondary_features, and shape — fall back on the textbook
   morphology of the given disease and describe the most clinically typical
   finding.

5. NEVER return null, "unclear", "not visible", or "unknown". Always commit to
   a specific, decisive clinical description.

6. Use standard dermatology terminology. Do NOT describe skin color or
   Fitzpatrick tone — that is provided separately.

7. DO NOT use self-defining terms like "psoriasiform", "verrucous",
   "acneiform", or "herpetiform". Describe actual visual features instead.

8. Use precise COLOR terms per dermatology convention:
   - erythematous = red, blanches
   - violaceous = purple
   - purpuric = red/purple, non-blanching
   - dusky = dark purple/grey, suggests necrosis

9. secondary_features uses the ABCS framework: Atrophy/scarring,
   Breakage (erosion/ulcer/fissure), Crap on skin (scale/crust/eschar),
   Scratching (lichenification/excoriation).

10. sd_prompt_body — THE MOST IMPORTANT FIELD. This is the actual text that
    will be fed to Stable Diffusion 1.5. Write it as flowing clinical prose,
    NOT a keyword list. Describe compound visual relationships between features
    (e.g. "silvery scale overlying dusky red plaques" not "silvery, scale,
    dusky, red, plaques"). Include the disease-specific morphology that makes
    this condition visually distinctive. 30-45 words, no skin tone. Focus on
    texture, color, surface quality, and material-like descriptors that an image
    generator can render. Do NOT include body part or Fitzpatrick type — those
    are prepended separately.

    EXAMPLES of good sd_prompt_body values:

    Psoriasis (plaques on elbow):
    "multiple well-circumscribed thickened plaques, striking silvery-white
    micaceous scale, dusky red-violet underlying color, shallow fissures
    within scale, post-inflammatory hyperpigmentation at lesion edges"

    Psoriasis (forearm):
    "dense coalescing plaques, thick adherent micaceous silvery scale
    obscuring red plaque color, natural body hair growing through lesions,
    large affected area, rough flaky texture"

    Psoriasis (nail dystrophy):
    "subungual hyperkeratosis, nail pitting on nail plate, onycholysis
    visible, thickened reddish-violet paronychial skin with micaceous scale"

    Seborrheic keratosis (trunk):
    "multiple stuck-on waxy papules and plaques, well-demarcated borders,
    verrucous surface with horn cysts, dark brown to tan color, greasy
    appearance"

    Note the pattern: compound phrases with adjective-noun clinical pairs,
    spatial relationships (scale overlying/within/obscuring), and rich
    texture words (micaceous, adherent, flaky, waxy, greasy).

11. Return ONLY the JSON object — no explanation, no markdown fences.
"""

SLOT_FALLBACKS = {
    "orientation":        "clinical view",
    "laterality":         "unspecified",
    "body_part":          "skin surface",
    "primary_lesion":     "skin lesion",
    "shape":              "round",
    "configuration":      "discrete",
    "lesion_color":       "erythematous",
    "texture":            "raised",
    "border_character":   "visible",
    "distribution":       "localized",
    "secondary_features": "surface changes",
    "background_skin":    "surrounding skin visible",
    "sd_prompt_body":     "",   # empty triggers slot-based fallback assembly
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── CSV loader ──────────────────────────────────────────────────────────────────

def load_dataset_items() -> pd.DataFrame:
    log.info(f"Loading dataset from {CSV_PATH}...")
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"{CSV_PATH} not found.")
    df = pd.read_csv(CSV_PATH)
    before = len(df)
    df = df[df["fst"].notna() & (df["fst"] != -1)].copy()
    log.info(f"Dropped {before - len(df)} rows with fst == -1 or null")
    df["fst_class"] = df["fst"].apply(lambda x: f"FP{int(x)}")
    df = df.rename(columns={"label": "disease_label", "nine_partition_label": "disease_group"})
    df["partition"] = df["partition"].str.strip().str.upper()
    df["gcs_uri"] = df["md5hash"].apply(lambda h: f"gs://{IMAGE_BUCKET}/{IMAGE_PREFIX}{h}.jpg")
    return df


# ── GCS helpers ─────────────────────────────────────────────────────────────────

def download_image_bytes(blob_name: str) -> bytes | None:
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(IMAGE_BUCKET)
        return bucket.blob(blob_name).download_as_bytes()
    except Exception as e:
        log.warning(f"GCS download failed for {blob_name}: {e}")
        return None


# ── Image preprocessing ──────────────────────────────────────────────────────────

def crop_watermark(img: np.ndarray) -> np.ndarray:
    h = img.shape[0]
    return img[:int(h * (1 - WATERMARK_CROP_PCT)), :]


def qc_source_image(img_bgr: np.ndarray) -> tuple[bool, str]:
    h, w = img_bgr.shape[:2]
    if min(h, w) < MIN_DIMENSION:
        return False, f"too_small_{w}x{h}"
    return True, "passed"


def resize_for_gemini(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = CLAUDE_RESIZE_PX / max(h, w)
    if scale >= 1.0:
        return img_bgr
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ── Gemini Vision call ───────────────────────────────────────────────────────────

async def call_gemini(
    client: genai.Client,
    img_bytes: bytes,
    disease: str,
    disease_group: str,
    md5hash: str,
    sem: asyncio.Semaphore,
) -> dict | None:
    async with sem:
        for attempt in range(2):
            try:
                text_part = (
                    f"Disease label (ground truth): {disease}\n"
                    f"Disease category (ground truth): {disease_group}\n\n"
                    + GEMINI_PROMPT
                )
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=[
                            genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                            text_part,
                        ],
                        config=genai_types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=1500,
                            thinking_config=genai_types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
                        ),
                    )
                )
                raw = response.text.strip()
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1] if len(parts) > 1 else raw
                    if raw.startswith("json"):
                        raw = raw[4:]
                return json.loads(raw.strip())
            except json.JSONDecodeError as e:
                log.warning(f"JSON parse error {md5hash} attempt {attempt+1}: {e}")
                if attempt == 1:
                    return None
            except Exception as e:
                log.warning(f"Gemini API error {md5hash} attempt {attempt+1}: {e}")
                if attempt == 1:
                    return None
                match = re.search(r'retry in (\d+(?:\.\d+)?)s', str(e), re.IGNORECASE)
                wait = float(match.group(1)) + 1 if match else 30
                log.info(f"Waiting {wait:.0f}s before retry...")
                await asyncio.sleep(wait)
    return None


# ── Prompt assembly ──────────────────────────────────────────────────────────────

def get_slot(gemini_json: dict, key: str) -> str:
    val = gemini_json.get(key)
    return str(val) if val else SLOT_FALLBACKS[key]


_clip_tok = None

def _get_clip_tokenizer():
    global _clip_tok
    if _clip_tok is None:
        from transformers import CLIPTokenizer
        _clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_tok


EMPTY_SLOT_VALUES = {
    "none", "none visible", "not visible", "absent", "unspecified",
    "null", "unknown", "n/a", "na", "nil",
}


def is_empty_slot(val: str) -> bool:
    return val.strip().lower() in EMPTY_SLOT_VALUES


def clip_truncate(prompt: str, max_tokens: int = 75) -> str:
    """
    Truncate to SD 1.5's real CLIP token budget (77 total - 2 special tokens).
    Drops trailing comma-clauses from the original string — never calls
    tokenizer.decode(), which would BPE-normalize and spoil formatting.
    """
    tok = _get_clip_tokenizer()
    if len(tok.encode(prompt, add_special_tokens=False)) <= max_tokens:
        return prompt
    log.warning(f"Prompt {len(tok.encode(prompt, add_special_tokens=False))} tokens > {max_tokens}, truncating")
    clauses = [c.strip() for c in prompt.split(",") if c.strip()]
    while clauses and len(tok.encode(", ".join(clauses), add_special_tokens=False)) > max_tokens:
        clauses.pop()
    return ", ".join(clauses)


def _fallback_slot_assembly(gemini_json: dict, disease: str, fst: str) -> str:
    """Slot-based prompt assembly — used when sd_prompt_body is missing."""
    s = lambda k: get_slot(gemini_json, k)
    parts = [
        f"clinical close-up of {s('body_part')} on {fst}",
        f"{disease} with {s('primary_lesion')}",
    ]
    color, tex = s("lesion_color"), s("texture")
    if not is_empty_slot(color) and not is_empty_slot(tex):
        parts.append(f"{color} {tex}")
    elif not is_empty_slot(color):
        parts.append(color)
    elif not is_empty_slot(tex):
        parts.append(tex)
    border = s("border_character")
    if not is_empty_slot(border):
        parts.append(f"{border} borders")
    sec = s("secondary_features")
    if not is_empty_slot(sec):
        parts.append(sec)
    parts.append("photorealistic DSLR macro")
    return ", ".join(parts)


def assemble_prompt(gemini_json: dict, disease: str, fst_class: str) -> str:
    """
    Build the SD prompt using Gemini's free-form sd_prompt_body when available,
    falling back to mechanical slot assembly otherwise.

    Template:
      "clinical close-up of {body_part} on {FST}, {disease} presenting
       {sd_prompt_body}, photorealistic DSLR macro"
    """
    fst = FST_DESCRIPTIONS[fst_class]
    body_part = get_slot(gemini_json, "body_part")
    prompt_body = gemini_json.get("sd_prompt_body", "")

    if not prompt_body or not prompt_body.strip():
        log.info("sd_prompt_body missing, using slot-based fallback")
        return clip_truncate(_fallback_slot_assembly(gemini_json, disease, fst))

    prompt = (
        f"clinical close-up of {body_part} on {fst}, "
        f"{disease} presenting {prompt_body}, "
        f"photorealistic DSLR macro"
    )
    return clip_truncate(prompt)


# ── Per-image processing ─────────────────────────────────────────────────────────

async def process_image(
    row: dict,
    client: genai.Client,
    sem: asyncio.Semaphore,
    cache: dict,
    rejected: list,
    cache_lock: asyncio.Lock,
) -> list[dict] | None:

    md5hash     = row["md5hash"]
    fst_class   = row.get("fst_class", "UNKNOWN")
    disease     = row.get("disease_label", "unspecified")
    disease_grp = row.get("disease_group", "unspecified")
    partition   = row.get("partition", "TRAIN")
    source_uri  = f"gs://{IMAGE_BUCKET}/{IMAGE_PREFIX}{md5hash}.jpg"

    loop = asyncio.get_event_loop()
    raw_bytes = await loop.run_in_executor(
        None, download_image_bytes, f"{IMAGE_PREFIX}{md5hash}.jpg"
    )
    if raw_bytes is None:
        rejected.append({"md5hash": md5hash, "reason": "gcs_download_failed"})
        return None

    img_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        rejected.append({"md5hash": md5hash, "reason": "decode_failed"})
        return None

    img_bgr = crop_watermark(img_bgr)
    h, w = img_bgr.shape[:2]

    passed, reason = qc_source_image(img_bgr)
    if not passed:
        rejected.append({"md5hash": md5hash, "fst_class": fst_class, "reason": reason})
        return None

    if md5hash in cache:
        gemini_json = cache[md5hash]
    else:
        resized_bgr   = resize_for_gemini(img_bgr)
        _, buf        = cv2.imencode(".jpg", resized_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        resized_bytes = buf.tobytes()

        gemini_json = await call_gemini(client, resized_bytes, disease, disease_grp, md5hash, sem)
        if gemini_json is None:
            rejected.append({"md5hash": md5hash, "reason": "gemini_api_failed"})
            return None
        cache[md5hash] = gemini_json
        async with cache_lock:
            with open(PROMPTS_CACHE, "a") as f:
                f.write(json.dumps({"md5hash": md5hash, **gemini_json}) + "\n")

    prompt = assemble_prompt(gemini_json, disease, fst_class)

    rows = []
    for v in range(VARIANTS_PER):
        seed = BASE_SEED + (hash(md5hash) % 10000) + v
        rows.append({
            "md5hash":                md5hash,
            "fst_class":              fst_class,
            "disease_label":          disease,
            "disease_group":          disease_grp,
            "partition":              partition,
            "variant_index":          v,
            "source_gcs_uri":         source_uri,
            "synthetic_gcs_uri":      f"gs://fp17kc-synthetic/{fst_class}/{md5hash}_v{v}.jpg",
            "assembled_prompt":       prompt,
            "negative_prompt":        UNIVERSAL_NEGATIVE,
            "sd_prompt_body":         gemini_json.get("sd_prompt_body", ""),
            "orientation":            get_slot(gemini_json, "orientation"),
            "laterality":             get_slot(gemini_json, "laterality"),
            "body_part":              get_slot(gemini_json, "body_part"),
            "primary_lesion":         get_slot(gemini_json, "primary_lesion"),
            "shape":                  get_slot(gemini_json, "shape"),
            "configuration":          get_slot(gemini_json, "configuration"),
            "lesion_color":           get_slot(gemini_json, "lesion_color"),
            "texture":                get_slot(gemini_json, "texture"),
            "border_character":       get_slot(gemini_json, "border_character"),
            "distribution":           get_slot(gemini_json, "distribution"),
            "secondary_features":     get_slot(gemini_json, "secondary_features"),
            "background_skin":        get_slot(gemini_json, "background_skin"),
            "vision_model":           GEMINI_MODEL,
            "watermark_cropped":      True,
            "source_width":           w,
            "source_height":          h,
            "source_passed_qc":       True,
            "strength":               SD_STRENGTH,
            "guidance_scale":         SD_GUIDANCE_SCALE,
            "controlnet_scale":       SD_CONTROLNET_SCALE,
            "steps":                  SD_STEPS,
            "seed":                   seed,
            "output_laplacian":       None,
            "output_ssim":            None,
            "output_skin_ratio":      None,
            "passed_qc":              None,
            "rejection_reason":       None,
            "automl_split":           "TRAIN",
            "generated_at":           None,
            "pipeline_version":       "v1.0",
            "manifest_created_at":    datetime.utcnow().isoformat(),
        })
    return rows


# ── Main ────────────────────────────────────────────────────────────────────────

async def main():
    log.info("=== Fitzpatrick Manifest Generator — TEST RUN ===")
    log.info(f"Project:        {PROJECT_ID}")
    log.info(f"Vision model:   {GEMINI_MODEL}")
    log.info(f"Test limit:     {TEST_LIMIT} images")

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=VERTEX_LOCATION,
    )
    log.info(f"Vertex AI client initialized ({PROJECT_ID} / {VERTEX_LOCATION})")

    df = load_dataset_items()
    df_filtered = df[
        df["fst_class"].isin(TARGET_FST) &
        (df["partition"] == "TRAIN")
    ].copy()

    # Take a balanced sample across FP5/FP6
    samples = []
    per_class = TEST_LIMIT // len(TARGET_FST)
    for fst in TARGET_FST:
        samples.append(df_filtered[df_filtered["fst_class"] == fst].head(per_class))
    df_test = pd.concat(samples).reset_index(drop=True)

    log.info(f"Test images: {len(df_test)} ({dict(df_test['fst_class'].value_counts())})")

    Path(PROMPTS_CACHE).touch(exist_ok=True)
    cache = {}
    with open(PROMPTS_CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[entry["md5hash"]] = {k: v for k, v in entry.items() if k != "md5hash"}
            except json.JSONDecodeError:
                log.warning(f"Skipping malformed cache line: {line[:80]}")
    log.info(f"Loaded {len(cache)} cached responses")

    sem        = asyncio.Semaphore(MAX_CONCURRENT)
    cache_lock = asyncio.Lock()
    rejected   = []

    tasks = [
        process_image(row.to_dict(), client, sem, cache, rejected, cache_lock)
        for _, row in df_test.iterrows()
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Test run")

    all_rows = [r for result in results if result for r in result]

    if all_rows:
        manifest_df = pd.DataFrame(all_rows)
        manifest_df.to_csv(OUTPUT_MANIFEST, index=False)
        log.info(f"\nTest manifest saved: {OUTPUT_MANIFEST}")
        log.info(f"  Rows:             {len(manifest_df)}")
        log.info(f"  Unique sources:   {manifest_df['md5hash'].nunique()}")
        log.info(f"\nSample prompt:\n  {manifest_df['assembled_prompt'].iloc[0]}")
    else:
        log.warning("No rows generated — check test_rejected.jsonl")

    if rejected:
        with open(REJECTED_LOG, "w") as f:
            for entry in rejected:
                f.write(json.dumps(entry) + "\n")
        log.info(f"Rejected: {len(rejected)} — see {REJECTED_LOG}")


if __name__ == "__main__":
    asyncio.run(main())
