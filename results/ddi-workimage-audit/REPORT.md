# Mask-exclusive workimage audit

## Result

- Images analyzed: 19
- Median mask-exclusive pixel ITA: 12.21°
- Median neural ITA after neutral filling: -5.82°
- Median absolute difference between methods: 25.50°
- Images recommended for review: 13/19

Pure-black pixels and black mask regions were excluded from pixel ITA. Before neural inference, excluded pixels were replaced with each image's median retained-skin color, so black was not supplied to the released model.

## Per-image results

| ID | Valid skin | Pixel ITA | Neural ITA | FST | Fold agreement | Review |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 000061 | 99.5% | -2.8 | -14.0 | 5 | 60% | yes |
| 000062 | 98.6% | 6.0 | 12.8 | 4 | 100% | no |
| 000083 | 99.0% | 43.1 | -21.0 | 6 | 100% | yes |
| 000100 | 68.7% | 3.8 | -9.9 | 4 | 80% | no |
| 000101 | 42.0% | -39.5 | -10.2 | 6 | 100% | yes |
| 000138 | 68.4% | 34.5 | -8.0 | 5 | 100% | yes |
| 000230 | 55.3% | -20.7 | -31.7 | 5 | 100% | yes |
| 000271 | 44.2% | 38.6 | 0.8 | 4 | 80% | yes |
| 000277 | 63.1% | 12.2 | 7.3 | 4 | 80% | no |
| 000287 | 68.6% | 32.8 | 7.3 | 3 | 100% | yes |
| 000289 | 47.2% | -11.3 | 4.6 | 4 | 100% | yes |
| 000293 | 64.4% | -3.0 | -5.8 | 4 | 100% | no |
| 000300 | 96.1% | 60.6 | 15.1 | 3 | 100% | yes |
| 000301 | 97.9% | 62.9 | 0.8 | 4 | 80% | yes |
| 000318 | 66.6% | -21.6 | -10.6 | 1 | 100% | no |
| 000347 | 45.2% | 37.6 | -11.0 | 4 | 100% | yes |
| 000348 | 98.3% | 47.9 | 9.7 | 3 | 100% | yes |
| 000426 | 99.4% | 27.4 | -1.6 | 5 | 100% | yes |
| 000434 | 67.2% | 3.8 | -6.9 | 3 | 100% | no |

## Interpretation limits

- Mask-exclusive pixel ITA is the primary result.
- Neural ITA and Fitzpatrick are secondary experimental outputs from a model trained for dermoscopy; several source images appear to be clinical photographs.
- Median filling removes black as an input signal but is outside the released model's original validation protocol.
- Review thresholds are operational flags, not calibrated clinical confidence.
- These outputs estimate image appearance, not race, ethnicity, diagnosis, or patient identity.
