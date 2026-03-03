# Sack Counting using YOLO and Ultralytics

## Problem Statement

The goal of this project is to count the number of jute/gunny sacks being loaded into a truck using video analytics.


---

## Approach

### 1. Initial Approach – Person-Based Counting

At first, I tried detecting the `person` class using a pretrained YOLO model and used Ultralytics' built-in `ObjectCounter` to count people crossing a defined line.

**Issues Faced:**
- Workers overlapped heavily near the truck ramp
- Tracker frequently lost object IDs due to occlusion
- Random people moving in the background were counted
- Counting people did not accurately represent sacks being loaded

This approach was unreliable, so I shifted to detecting sacks directly.

### 2. Second Attempt – Small Custom Dataset

I extracted frames from the videos and created a small custom dataset (~33 images) with annotated sacks. I trained YOLO using transfer learning on this dataset.

**Result:**
- The model overfitted quickly
- Detection was unstable in some videos
- Performance varied depending on camera angle
- The dataset was too small and too specific

### 3. Third Attempt – Larger Roboflow Dataset (~1.8k Images)

I trained a model using a larger external sack dataset. The validation metrics were high, but the model did not perform well on my target videos.

This happened due to domain shift — the external dataset had different lighting, perspective, and environment compared to the truck loading videos.

### 4. Final Approach – Dataset Merging (Domain Adaptation)

To solve the domain mismatch, I merged:
- The 1.8k external sack dataset
- My 33 truck-specific annotated frames

This allowed the model to learn both:
- General sack features
- Scene-specific characteristics from the truck videos

The merged dataset significantly improved detection stability.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Model | yolo26n.pt (pretrained) |
| Transfer Learning | Yes |
| Image Size | 640 |
| Epochs | 40 |
| Batch Size | 16 |
| Tracker | ByteTrack (via Ultralytics) |
| Counting | Ultralytics ObjectCounter |

Training was done on **Kaggle** (GPU enabled). The training notebook is included in the `src/` folder.

---

## Results

- **Video 2** (stable camera) produced the most accurate counts
- **Video 1 and 3** were less stable due to:
  - Camera movement
  - Background activity
  - Workers moving randomly

The final model detects sacks reliably in most frames. Counting accuracy depends on scene stability and tracking consistency.

Training curves, loss graphs, and prediction samples are available in the `outputs/` folder.

![results_image](C:\Projects\obj-detection-counting\output_5\kaggle\working\custom_sack_detection\results.png)

---

## Tools Used

- Python
- Ultralytics YOLO
- OpenCV
- Roboflow (annotation)
- Kaggle (training environment)

---

## Key Learnings

- High validation mAP does not guarantee real-world performance
- Domain-specific data is very important
- Small datasets can cause overfitting quickly
- Tracking stability affects counting accuracy
- Scene complexity (occlusion, overlap) makes counting harder than simple object detection

---

## Repository Structure

```
├── src/
│   ├── odc_ultra.py
│   └── yolo-fine-tuning.ipynb
│
├── output_5/
│   ├── training_results/
│   └── prediction_samples/
│
├── sack-1/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── requirements.txt
└── README.md
```

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run sack detection and counting
python odc_ultra.py
```
