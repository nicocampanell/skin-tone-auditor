# Skin-tone representation screening audit

## Scope

This audit applies the five-fold Fitzpatrick and colorimeter-supervised Lab/ITA ensembles released with `nn_colorimetry_dermatoscopy` to 16 dermoscopic images. The source dataset has diagnosis-folder labels but no skin-tone ground truth, patient IDs, acquisition metadata, or rateability labels. Predictions therefore describe the model's estimated image distribution; they do not establish demographic identity or equal clinical performance.

## Screening findings

- Predicted FST V-VI coverage is only 0.00%; darker skin appears underrepresented.
- 18.75% of images have high ensemble disagreement and should be reviewed.

- Median predicted ITA: 28.96 degrees.
- Review recommended by the unvalidated ensemble-disagreement heuristic: 3/16 (18.75%).
- Exact duplicate image rows: 0; exact duplicates spanning train/test: 0.

### Predicted Fitzpatrick groups

| category | count | percent |
| --- | --- | --- |
| I-II | 9 | 56.25 |
| III-IV | 7 | 43.75 |

### Predicted ITA categories

| category | count | percent |
| --- | --- | --- |
| dark | 0 | 0.0 |
| brown | 4 | 25.0 |
| tan | 3 | 18.75 |
| intermediate | 7 | 43.75 |
| light | 1 | 6.25 |
| very_light | 1 | 6.25 |

### Diagnosis-folder comparison

| diagnosis_folder | images | FST_V_VI_percent | median_ITA | review_percent |
| --- | --- | --- | --- | --- |
| benign | 16 | 0.0 | 28.96 | 18.75 |

## Interpretation limits

- Fitzpatrick is presented only as compatibility metadata; ITA is the preferred continuous estimate.
- The model was validated on a small MSKCC dermoscopy/colorimeter cohort, not on this Kaggle collection.
- Lesions, black borders, hair, lighting, and acquisition devices can influence estimates.
- Folder names are diagnosis labels, not patient demographics; subgroup differences may reflect lesion appearance or source composition.
- Ensemble agreement is not calibrated confidence and does not replace a dermatologist rateability review.
- A defensible equality claim requires representative patient-level ground truth and downstream diagnostic performance by tone, neither of which this dataset supplies.
