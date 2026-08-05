# aircraft_skin_defects Manual Review for Thesis Writing

## 1. Problem Diagnosis

The manual review of the main 5-class dataset shows that the dominant source of performance instability is not pure missing annotation. Instead, the main problems are label-granularity inconsistency and class-definition ambiguity.

The most typical issues are:

1. `missing-head` contains many tiny boxes attached to rivets, holes, edges, characters, or normal structural points, which makes the class boundary highly unstable.
2. `paint-off` is often fragmented into many small adjacent boxes, so one visually continuous damaged region may be annotated as multiple tiny instances.
3. Tiny-box noise is dense in structurally complex regions, which increases the risk of false-positive learning for small defects.

These observations explain why the dataset is more sensitive to annotation noise and small-object ambiguity than a clean benchmark-style dataset.

## 2. Method Choice

Because the main problem is data quality and class-distribution mismatch rather than insufficient backbone capacity, the thesis should prioritize data-centric correction over aggressive structural modification.

The practical design choice is:

- keep the official YOLO11n baseline as the reference detector;
- avoid making the final conclusion depend on a complex wavelet or NWD structure;
- use offline Copy-Paste to strengthen the weak `scratch` class under the original validation and test splits.

This method choice is consistent with the manual diagnosis: when a class is both weak and under-supported, targeted data enhancement is more controllable than adding extra structural complexity on top of noisy labels.

## 3. Experimental Verification

The main-dataset experiments support this diagnosis-method alignment.

- `B0` provides a stable reference baseline.
- `Exp_A` (baseline + offline Copy-Paste) achieves the best overall result on the main dataset and delivers the strongest `scratch` improvement.
- `Exp_B` and `Exp_C` show that neck-side wavelet refinement does not produce a reliable gain, even when combined with Copy-Paste.

Therefore, the validated conclusion is not "the new structure is universally better," but rather:

**for this aircraft skin defect dataset, a defect-oriented data enhancement strategy is more effective and more reliable than the tested structural modifications.**

## 4. Thesis Wording Suggestion

In the thesis, this section can be written as:

"Manual inspection indicates that the main dataset suffers from class ambiguity, excessive tiny-box fragmentation, and noisy `missing-head` annotations. Based on this diagnosis, the study prioritizes data-centric intervention and introduces an offline Copy-Paste strategy targeted at the weak `scratch` category. Experimental results show that this targeted data enhancement outperforms the tested structural modification branches and yields the best final model on the main dataset."
