# Stable Diffusion augmentation for darker skin tones

Fitzpatrick types V and VI are badly under-represented in dermatology datasets; cleaned Fitzpatrick17k has only 369 type VI images. This stage generates realistic synthetic images of skin conditions on darker skin to rebalance training data. Synthetic images are used for training only, never for validation or test.

| File | What it does |
|---|---|
| [`Stable Diffusion Code.ipynb`](Stable%20Diffusion%20Code.ipynb) | Colab notebook with prompt and pipeline iterations v2–v11 on psoriasis images. It moves from plain text-to-image ControlNet, through img2img, to **ControlNet SoftEdge (HED) + img2img**, which keeps the lesion's geometry while changing skin tone. Later versions relocate lesions to new body parts with skin-tone prompt templates. |
| [`generate_manifest.py`](generate_manifest.py) | Phase 2 of the production pipeline. Reads FP5/FP6 training images, downloads them from Cloud Storage, crops watermarks and runs quality checks. It then calls **Gemini 2.5 Flash vision** asynchronously to extract structured clinical descriptors (lesion type, shape, color, border, distribution…). Finally it assembles CLIP-length-safe Stable Diffusion prompts and writes a full manifest before any GPU time is spent. |
| [`generate_manifest_test.py`](generate_manifest_test.py) | The same pipeline limited to 30 images, writing to separate test files. |
| [`outputs/`](outputs) | Test-run manifest, Gemini prompt cache, and the log of source images that failed quality checks. |

Example manifest row (from [`outputs/test_manifest.csv`](outputs/test_manifest.csv)):

> *clinical close-up of upper arm on Fitzpatrick type V, brown skin, deep tan tone, lupus erythematosus presenting numerous discrete violaceous papules, round to ovoid in shape, slightly raised from the skin surface, scattered across the arm and neck, with well-defined borders., photorealistic DSLR macro*

Generated images went through dermatologist "real vs. synthetic" review panels. The notes from those panels (artificial pink tones, implausible scab placement, images too clean to be real clinical photos) fed back into the prompt and color protocol.
