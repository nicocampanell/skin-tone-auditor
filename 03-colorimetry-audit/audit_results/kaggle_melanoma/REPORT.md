# Skin-tone representation screening audit

## Scope

This audit applies the five-fold Fitzpatrick and colorimeter-supervised Lab/ITA ensembles released with `nn_colorimetry_dermatoscopy` to 10,605 dermoscopic images. The source dataset has diagnosis-folder labels but no skin-tone ground truth, patient IDs, acquisition metadata, or rateability labels. Predictions therefore describe the model's estimated image distribution; they do not establish demographic identity or equal clinical performance.

## Screening findings

- Predicted FST V-VI coverage is only 3.01%; darker skin appears underrepresented.
- 28.13% of images have high ensemble disagreement and should be reviewed.
- Found 14 exact duplicate images spanning train and test folders.

- Median predicted ITA: 23.23 degrees.
- Review recommended by the unvalidated ensemble-disagreement heuristic: 2,983/10,605 (28.13%).
- Exact duplicate image rows: 44; exact duplicates spanning train/test: 14.

### Predicted Fitzpatrick groups

| category | count | percent |
| --- | --- | --- |
| I-II | 5537 | 52.21 |
| III-IV | 4749 | 44.78 |
| V-VI | 319 | 3.01 |

### Predicted ITA categories

| category | count | percent |
| --- | --- | --- |
| dark | 19 | 0.18 |
| brown | 2679 | 25.26 |
| tan | 3778 | 35.62 |
| intermediate | 2799 | 26.39 |
| light | 1238 | 11.67 |
| very_light | 92 | 0.87 |

### Diagnosis-folder comparison

| diagnosis_folder | images | FST_V_VI_percent | median_ITA | review_percent |
| --- | --- | --- | --- | --- |
| benign | 5500 | 0.78 | 25.2 | 16.64 |
| malignant | 5105 | 5.41 | 20.16 | 40.51 |

## Interpretation limits

- Fitzpatrick is presented only as compatibility metadata; ITA is the preferred continuous estimate.
- The model was validated on a small MSKCC dermoscopy/colorimeter cohort, not on this Kaggle collection.
- Lesions, black borders, hair, lighting, and acquisition devices can influence estimates.
- Folder names are diagnosis labels, not patient demographics; subgroup differences may reflect lesion appearance or source composition.
- Ensemble agreement is not calibrated confidence and does not replace a dermatologist rateability review.
- A defensible equality claim requires representative patient-level ground truth and downstream diagnostic performance by tone, neither of which this dataset supplies.
