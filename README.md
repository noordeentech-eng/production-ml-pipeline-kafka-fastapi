# Automated Quality Control with Transfer Learning
### Manufacturing Defect Detection — Discovery-to-Action (DTA) Strategy

An end-to-end transfer-learning pipeline that adapts a pre-trained **ResNet-50** CNN to classify production-line images as **`good`** or **`defect`**, and translates the model's output into automated **robotic arm sorting logic**.

---

## 1. Project Structure

```
.
├── transfer_learning_quality_control.ipynb   # Main notebook (Discovery → Technical → Action)
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
└── dataset/
    ├── train/
    │   ├── good/        # 60 synthetic "clean surface" images
    │   └── defect/       # 60 synthetic "scratch/dent/crack" images
    ├── test/
    │   ├── good/        # 20 held-out images
    │   └── defect/       # 20 held-out images
    └── new_images/        # 5 "unseen" images for the inference demo
```

> **About the dataset:** This repo ships with a small, **procedurally generated synthetic dataset** (brushed-metal surfaces, some clean, some with a simulated scratch/dent/crack) so the notebook runs end-to-end with zero external downloads. If `dataset/` is deleted, the notebook regenerates it automatically. **For a real deployment, replace `dataset/train/<class>` and `dataset/test/<class>` with your own labeled product photos** — the folder structure (`good/` and `defect/` subfolders) is all that's required; the rest of the pipeline is unchanged.

---

## 2. How to Run

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `transfer_learning_quality_control.ipynb` in Jupyter, JupyterLab, VS Code, or **Google Colab** (recommended for free GPU access and guaranteed internet access for the ImageNet pre-trained weights).
4. Run all cells top to bottom.

**Note on internet access:** `ResNet50(weights="imagenet", ...)` downloads ~98MB of pre-trained weights on first use. The notebook will automatically fall back to `weights=None` (random initialization) if it cannot reach the internet, so it will still execute — but the transfer-learning benefits (and meaningful accuracy) require the ImageNet weights, so run it somewhere with internet access (e.g., Colab) for real results.

---

## 3. Dataset Preparation (Discovery Phase)

- **Resizing & normalization:** All images are resized to **224×224×3** (ResNet-50's expected input) and passed through `tensorflow.keras.applications.resnet50.preprocess_input`, which applies the same channel-wise mean subtraction the backbone saw during ImageNet pre-training.
- **Augmentation (training set only):** `ImageDataGenerator` applies horizontal flips, ±20° rotation, 15% zoom, and small width/height shifts — simulating the part-orientation and camera-alignment variation seen on a real production line, and reducing overfitting on a small dataset.
- **Train/validation/test split:** `dataset/train/` is split 80/20 into training and validation subsets via `validation_split`; `dataset/test/` is a fully held-out set used only for final evaluation.
- **Class index mapping:** Keras assigns class indices alphabetically — `defect = 0`, `good = 1`. The model's sigmoid output is therefore `P(good)`, and the notebook explicitly computes `defect_probability = 1 - P(good)` wherever a "defect probability" is reported.

---

## 4. Transfer Learning Workflow (Technical Phase — "Brain Swap")

| Step | Detail |
|---|---|
| **Backbone** | `ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))` — strips the original 1000-class ImageNet head, keeping only convolutional feature extraction. |
| **Freezing** | `base_model.trainable = False` — only the new head is trained initially, preserving pre-learned visual features and preventing overfitting/catastrophic forgetting on the small QC dataset. |
| **Custom head** | `GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)` |
| **Compile** | Adam (`lr=1e-4`), `binary_crossentropy` loss, tracked metrics: accuracy, precision, recall |
| **Training** | Up to 15 epochs with `EarlyStopping` (restores best weights) and `ReduceLROnPlateau` |
| **Optional fine-tuning** | A flagged, off-by-default section unfreezes the last ~10 backbone layers with `lr=1e-5` for further adaptation once a larger real dataset is available |

### Why `GlobalAveragePooling2D` over `Flatten`?

1. **Parameter reduction:** `Flatten` on a 7×7×2048 feature map → `Dense(128)` needs ~12.85M parameters; `GlobalAveragePooling2D` → `Dense(128)` needs only ~262K — roughly **49× fewer parameters**.
2. **Spatial awareness / translation invariance:** Defects (scratches, dents, cracks) can appear *anywhere* on a part. GAP averages each feature channel across all spatial positions, producing a representation that captures *whether* a pattern is present rather than *where* — a much better match for defect detection than Flatten's position-sensitive vector.
3. **Overfitting mitigation:** The drastic dimensionality reduction (100,352 → 2048) acts as an implicit regularizer, critical when labeled defect images are scarce.

---

## 5. Performance Evaluation (Action Phase)

The notebook produces, on the held-out test set:
- Training/validation **loss and accuracy curves** across all epochs.
- A full `sklearn.metrics.classification_report` with **precision, recall, and F1-score** for both `good` and `defect` classes.
- A **confusion matrix** visualization.

> Because the shipped dataset is synthetic and small, exact numbers will vary run to run and are not a substitute for evaluation on real production images — re-run on real data for production-grade metrics.

**Why recall on `defect` matters most:** missing a real defect (false negative) means a faulty part reaches the customer — far costlier than a false alarm that sends a good part for an extra manual check.

---

## 6. Factory Decision Logic (Action Phase)

For each new image, the notebook computes `defect_probability = 1 - P(good)` and maps it to a robotic-arm command via `factory_decision()`:

| `defect_probability` | Action | Robotic Arm Command |
|---|---|---|
| **≥ 0.85** | `REJECT` | Divert part to reject bin via pneumatic actuator; log part ID, timestamp, image, and score to the defect database. |
| **0.50 – 0.85** | `FLAG_FOR_REVIEW` | Route part to a manual-inspection buffer lane; **no automatic scrap** — awaits human QC decision. |
| **< 0.50** | `PASS` | Allow the part to continue on the main conveyor to packaging. |

The **0.85 reject threshold** is deliberately conservative: fully-automated, irreversible rejection only happens when the model is highly confident, while the 0.50–0.85 "gray zone" defers ambiguous parts to a human reviewer rather than guessing — reflecting the asymmetric cost of false negatives vs. false positives in QC.

---

## 7. Limitations & Next Steps for Real-World Deployment

- **Synthetic data only.** Re-validate on real product photographs covering real lighting, materials, and defect types before any production use.
- **Class imbalance.** Real defect rates are typically low (often <5%); use `class_weight`, focal loss, or targeted data collection so recall on `defect` doesn't collapse.
- **Multi-class defects.** Extend the single-sigmoid output to `softmax`/multi-label outputs if the line needs to distinguish defect *types* for root-cause analysis.
- **Threshold tuning.** Re-derive the 0.85 / 0.50 thresholds from a precision-recall curve on real validation data, driven by the relative cost of false negatives vs. false positives — and revisit periodically as the model is retrained.
- **Explainability.** Add Grad-CAM (or similar) visualizations so QC operators can see *where* the model detected a likely defect.
- **Edge deployment.** Convert to TensorFlow Lite / ONNX for low-latency inference on factory-floor hardware, and benchmark against conveyor line speed.
- **Integration.** Wrap `factory_decision()` in a small inference service that the robotic controller / PLC can call.
- **Monitoring & retraining loop.** Track the live `defect_probability` distribution and human-review override rate; log every reviewed part's true label to grow a real-world training set over time.

---

## 8. Tech Stack

- TensorFlow / Keras (ResNet-50, `ImageDataGenerator`)
- NumPy, Matplotlib
- scikit-learn (classification report, confusion matrix)
- Pillow (synthetic dataset generation)
