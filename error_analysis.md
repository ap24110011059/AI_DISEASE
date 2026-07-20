# Error Analysis

## False Negative Analysis

False negatives represent CKD patients incorrectly classified as healthy by the model. These cases are clinically important because they may delay treatment.

The following patterns were observed among the missed patients:

- Mild CKD symptoms.
- Near-normal hemoglobin values.
- Borderline serum creatinine levels.
- Overlapping characteristics with healthy individuals.
- Incomplete clinical measurements.

## Key Observation

The adaptive model performs well for severe CKD cases but occasionally struggles with early-stage disease where symptoms are less pronounced.

## Limitations

The current system may miss patients with subtle CKD indicators. Future work will incorporate additional biomarkers and larger datasets to improve sensitivity and reduce false negatives.