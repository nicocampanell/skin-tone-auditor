# Fitzpatrick17k skin-tone classifier

The project's first stage: classifying skin tone directly from dermatology photos in Fitzpatrick17k.

| File | What it does |
|---|---|
| [`skin_tone_classifier.py`](skin_tone_classifier.py) | Lesion-robust binary classifier, Fitzpatrick I–III vs IV–VI. EfficientNet-B3 with a ConvNeXt-Tiny ensemble, class-balanced sampling, and optional Grad-CAM attention regularization so the model looks at skin rather than the lesion. Has `train`, `eval` and `predict` modes. |
| [`download_images.py`](download_images.py) | Parallel, resumable downloader for the Fitzpatrick17k image URLs. Saves each image as `<md5hash>.jpg`, with an option to keep only QC-passed images. |
| [`script17kc.ipynb`](script17kc.ipynb) | Prepares the cleaned Fitzpatrick17k-C labels and predefined train/val/test splits for a six-class Vertex AI AutoML model. |

```bash
pip install torch torchvision timm albumentations opencv-python scikit-learn pandas numpy tqdm
python skin_tone_classifier.py --mode train
python skin_tone_classifier.py --mode predict --image path/to/image.jpg
```

**What we learned:** a coarse light / medium / dark split reached PR AUC 0.946 on the cleaned 10k subset, but six-class Fitzpatrick only reached 0.774. Per-class performance tracked class size almost exactly, because types V and VI are scarce. That result led to the synthetic-data work in [`02-stable-diffusion-augmentation`](../02-stable-diffusion-augmentation).
