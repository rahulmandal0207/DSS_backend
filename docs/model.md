# Training the classifier

The classifier is optional. DSS runs on its rules without it, and no trained model or dataset ships
with this repository.

## 1. Prepare images

Sort images into a folder per label, with a subfolder per person:

```text
dataset/
├── drowsy/
│   ├── person_01/
│   └── person_02/
└── notdrowsy/
    ├── person_01/
    └── person_03/
```

Labels are `drowsy`, and `notdrowsy` or `alert`. The person folders matter: see the next section.

## 2. Extract features

```bash
dss extract --input dataset --output data/features.csv
```

This writes one row per usable image, with the columns `subject`, `drowsy`, `ear_left`, `ear_right`,
`mar`, `pitch`, `yaw` and `roll`. Images without a usable face are skipped and counted in the log.

## 3. Train

```bash
dss train --data data/features.csv --output models/drowsiness.joblib
```

This fits a scaler and an SVM with calibrated probabilities, then writes the model and a
`drowsiness.metrics.json` report beside it. Add `--kernel rbf` for a non-linear boundary.

## Reading the results

**Check `split` in the report first.**

- `by_subject` means the test set holds people the model never saw. This is the honest number, and
  the one to quote.
- `random` means no subject folders were available. Neighbouring video frames are nearly identical,
  so random splitting puts near-copies of the test set into the training set, and the score can look
  excellent while the model has learned little. Do not quote it as accuracy.

For a safety problem, recall on the drowsy class matters more than accuracy: a false alarm is an
annoyance, while a miss is the failure the system exists to prevent. The confusion matrix in the
report is laid out as `[[alert→alert, alert→drowsy], [drowsy→alert, drowsy→drowsy]]`.

## 4. Enable it

```bash
DSS_MODEL_PATH=models/drowsiness.joblib dss serve
```

The model's score is averaged over `DSS_MODEL_SMOOTHING_S` and raises a warning above
`DSS_MODEL_DROWSY_PROBABILITY`. It cannot clear an alert raised by a rule. The dashboard's status
JSON reports the score as `model_probability`.

Model files are pickles, and loading one runs code from it, so only load models you trained.

## Limits

The classifier sees single frames, so it cannot perceive duration: the temporal rules remain the
stronger signal. Its head pose features are raw angles without the driver's baseline, so train and
run with similar camera placement. Features extracted before version 1.0.0 are not compatible,
because the eye and mouth ratios were redefined. See the [changelog](../CHANGELOG.md).
