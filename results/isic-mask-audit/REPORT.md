# Mask-exclusive workimage audit

## Result

- Images analyzed: 31
- Input mode: single_tiff_black_exclusion
- Median mask-exclusive pixel ITA: 69.31°
- Median neural ITA after neutral filling: 29.22°
- Median absolute difference between methods: 45.06°
- Images recommended for review: 30/31

Pure-black pixels and black mask regions were excluded from pixel ITA. Before neural inference, excluded pixels were replaced with each image's median retained-skin color, so black was not supplied to the released model.

## Per-image results

| ID | Valid skin | Pixel ITA | Neural ITA | FST | Fold agreement | Review |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ISIC_0276991 | 97.8% | 84.6 | 36.6 | 2 | 80% | yes |
| ISIC_0577187 | 94.0% | 58.6 | 39.4 | 4 | 80% | yes |
| ISIC_1447645 | 95.1% | 56.8 | 18.0 | 3 | 80% | yes |
| ISIC_1517860 | 99.1% | 24.5 | -4.3 | 5 | 60% | yes |
| ISIC_2756784 | 92.3% | 124.5 | 45.9 | 2 | 100% | yes |
| ISIC_2783387 | 98.1% | -68.4 | -32.5 | 5 | 100% | yes |
| ISIC_2805627 | 97.2% | 91.2 | 46.1 | 3 | 80% | yes |
| ISIC_2815015 | 98.8% | 13.4 | -34.9 | 6 | 100% | yes |
| ISIC_3041796 | 98.5% | 87.8 | -22.9 | 5 | 100% | yes |
| ISIC_3041908 | 94.4% | 46.3 | 33.0 | 4 | 60% | yes |
| ISIC_3374622 | 96.3% | 30.3 | 24.4 | 3 | 100% | no |
| ISIC_3661131 | 91.2% | 22.0 | 3.7 | 5 | 60% | yes |
| ISIC_3923841 | 98.4% | 3.7 | -53.4 | 6 | 100% | yes |
| ISIC_4234563 | 96.3% | 95.5 | 35.6 | 2 | 100% | yes |
| ISIC_4515429 | 97.4% | 120.3 | 58.2 | 3 | 80% | yes |
| ISIC_4733818 | 95.2% | 36.5 | 16.9 | 5 | 80% | yes |
| ISIC_4895408 | 94.6% | 169.0 | 30.3 | 3 | 100% | yes |
| ISIC_5387996 | 98.7% | 110.7 | 45.5 | 3 | 100% | yes |
| ISIC_5523792 | 95.2% | 69.3 | 27.8 | 3 | 100% | yes |
| ISIC_5887063 | 91.9% | 91.8 | 31.5 | 2 | 100% | yes |
| ISIC_6048055 | 98.7% | -90.6 | -36.5 | 5 | 100% | yes |
| ISIC_6099201 | 97.4% | 90.2 | 32.4 | 3 | 100% | yes |
| ISIC_6871733 | 80.3% | 54.3 | 13.2 | 4 | 100% | yes |
| ISIC_6962378 | 91.7% | 81.1 | 50.7 | 3 | 100% | yes |
| ISIC_8753288 | 98.0% | 88.2 | 42.3 | 3 | 100% | yes |
| ISIC_9373734 | 96.8% | 38.9 | 17.1 | 4 | 80% | yes |
| ISIC_9441129 | 97.7% | 85.9 | 33.5 | 2 | 100% | yes |
| ISIC_9631035 | 86.5% | 41.5 | 19.5 | 5 | 60% | yes |
| ISIC_9739516 | 98.3% | 75.0 | 46.6 | 3 | 60% | yes |
| ISIC_9990390 | 98.8% | 54.1 | 13.2 | 4 | 100% | yes |
| s | 88.5% | 92.4 | 29.2 | 2 | 80% | yes |

## Interpretation limits

- Mask-exclusive pixel ITA is the primary result.
- Neural ITA and Fitzpatrick are secondary experimental outputs from a model trained for dermoscopy; several source images appear to be clinical photographs.
- Median filling removes black as an input signal but is outside the released model's original validation protocol.
- Review thresholds are operational flags, not calibrated clinical confidence.
- These outputs estimate image appearance, not race, ethnicity, diagnosis, or patient identity.
