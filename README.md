# Ankit Shah Ph.D. Thesis - Carnegie Mellon University
This repository contains materials related to Ankit Shah's Ph.D. thesis at the Machine Learning and Signal Processing (MLSP) Group of Carnegie Mellon University (CMU). The aim is to consolidate research insights, findings, and code from the doctoral journey.

## Thesis Citation

> **Computational Audition with Imprecise Labels**
> Ankit Parag Shah
> Ph.D. Thesis, Carnegie Mellon University, 2024
> Advisor: Prof. Bhiksha Raj

**Links:** [KiltHub Repository](https://kilthub.cmu.edu/articles/thesis/Computational_Audition_with_Imprecise_Labels/28422542) | [PDF](http://cvis.cs.cmu.edu/cvis/docs/Ankit_Shah_Computational_Audition_with_Imprecise_labels_Thesis.pdf)

```bibtex
@phdthesis{shah2024computational,
  title={Computational Audition with Imprecise Labels},
  author={Shah, Ankit Parag},
  year={2024},
  school={Carnegie Mellon University},
  address={Pittsburgh, PA},
  note={Language Technologies Institute, School of Computer Science}
}
```

---

## Purpose
This repository aims to share research, publications, and supplementary materials aligned with the focus areas of the MLSP Group.

---

## Research Projects

### 1. Never Ending Learning of Sound (NELS)

**Code:** [`Never_Ending_Learning_of_Sound/`](./Never_Ending_Learning_of_Sound/) | **Original Repository:** [ankitshah009/Never_Ending_Learning_of_Sound](https://github.com/ankitshah009/Never_Ending_Learning_of_Sound)

**Description:**
NELS is a web-based intelligence system that continually searches the internet for sound samples and automatically learns their meanings, associations, and semantics. The project enables machines to continuously learn sounds that exist by crawling the entire web, helping them understand, sense, categorize, and model the relationships between different sounds.

**Key Features:**
- Continuous learning from the web about relations between sounds and language
- Self-improving sound recognition models over time
- Large-scale learning competency evaluation without references
- Integration with openSMILE toolkit for audio analysis, processing, and classification

**Technical Stack:**
- **Languages:** CSS, Python, HTML, Shell, MATLAB, C
- **Tools:** openSMILE (audio analysis), LibSVM (classification), PortAudio (sound recording)

**Project Structure:**
- `NEAL/` - Core project components
- `opensmile/` - Audio analysis toolkit configuration
- `scripts/` - Utility and processing scripts

**License:** Apache License 2.0

**Advisors:** Prof. Bhiksha Raj and Prof. Rita Singh (CMU MLSP Group)

**Publication:** [NELS - Never-Ending Learner of Sounds](https://ankitshah009.github.io/publication/2018-01-17-NELS-Never-Ending%20Learner%20of%20Sounds)

---

### 2. A Closer Look at Weak Label Learning for Audio Events (WALNet)

**Code:** [`WALNet-Weak_Label_Analysis/`](./WALNet-Weak_Label_Analysis/) | **Original Repository:** [ankitshah009/WALNet-Weak_Label_Analysis](https://github.com/ankitshah009/WALNet-Weak_Label_Analysis)

**Description:**
This project explores the challenges of large-scale Audio Event Detection (AED) using weakly labeled data through a CNN-based framework. The research examines how label density and label corruption affect performance, and compares mined web data versus manually labeled AudioSet data for training.

**Key Features:**
- Handles variable-length audio recordings without extensive preprocessing
- Adjustable segment sizes for secondary outputs
- Comprehensive analysis of weak label learning challenges
- Achieves 22.9 MAP on AudioSet (10s) and 83.5 MAP on ESC-50 dataset

**Technical Stack:**
- **Languages:** Python (99.8%), Shell (0.2%)
- **Features:** 128-dimensional MelSpectrogram features, embedding-level representations

**Project Structure:**
- `classifier/` - Classification models
- `download/` - Data download utilities
- `feature_extraction/` - Audio feature processing
- `model/` - Neural network architectures
- `lists/` - Dataset file lists

**License:** GPL-3.0

**Authors:** Ankit Shah, Anurag Kumar, Alexander G. Hauptmann, Bhiksha Raj

**Publication:** [arXiv:1804.09288](https://arxiv.org/abs/1804.09288)

---

### 3. DCASE 2017 Task 4 - Large-Scale Weakly Supervised Sound Event Detection for Smart Cars

**Code:** [`Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/`](./Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/) | **Original Repository:** [ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars](https://github.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars)

**Description:**
Official repository for DCASE 2017 Challenge Task 4, focusing on large-scale weakly supervised sound event detection for smart car applications. The task consists of audio tagging (AT) and sound event detection (SED) subtasks with 17 sound classes relevant to driving scenarios.

**Key Features:**
- Audio tagging and sound event detection subtasks
- 51,172 training audio segments, 488 testing segments
- 17 sound event classes (e.g., "Train horn", "Car")
- Multiprocessing support reducing download time by 60%
- Standardized audio formatting (44.1 kHz, mono, 16-bit)

**Technical Stack:**
- **Languages:** Python 2.7
- **Tools:** youtube-dl, pafy, sox, ffmpeg
- **Framework:** DCASE2017-baseline-system (submodule)

**Project Structure:**
- `DCASE2017-baseline-system/` - Baseline framework
- `evaluation/` - Subtask A evaluation scripts
- `TaskB_evaluation/` - Subtask B evaluation tools
- `groundtruth_release/` - Annotated ground truth labels

**Role:** Data & Annotations, Baseline & Metrics development

**Coordinators:** Benjamin Elizalde, Emmanuel Vincent, Bhiksha Raj

**Publication:** [DCASE 2017 Challenge](https://dcase.community/challenge2017/task-large-scale-sound-event-detection-results)

---

### 4. DCASE 2018 Task 4 - Large-Scale Weakly Labeled Semi-Supervised Sound Event Detection in Domestic Environments

**Code:** [`dcase2018_baseline/`](./dcase2018_baseline/) | **Original Repository:** [DCASE-REPO/dcase2018_baseline](https://github.com/DCASE-REPO/dcase2018_baseline)

**Description:**
Official baseline system for DCASE 2018 Challenge Task 4, evaluating systems for large-scale detection of sound events in domestic environments using weakly labeled data. The challenge explores exploiting unlabeled training data with weakly annotated sets for ambient assisted living applications.

**Key Features:**
- Semi-supervised approach using CRNN architecture
- Weakly labeled data (clip-level) combined with unlabeled data
- Domestic environment focus with smart home applications
- Macro-averaged event-based F-measure with 200ms collar

**Technical Stack:**
- **Languages:** Python
- **Framework:** dcase_util, Keras, TensorFlow
- **Tools:** YouTube audio extraction pipeline

**Dataset:**
- 1,578 weakly labeled training clips (2,244 class occurrences)
- 14,412 unlabeled in-domain training clips
- Subset of Google's AudioSet (632 sound event classes)

**Role:** Task Organizer - Code development, dataset development, paper reviews, system submissions, technical support

**Authors:** Romain Serizel, Nicolas Turpault, Hamid Eghbal-Zadeh, Ankit Parag Shah

**Publication:** [DCASE 2018 Workshop Proceedings](https://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection)

---

### 5. Learning Sound Events from Webly Labeled Data

**Code:** [`webly-labeled-sounds/`](./webly-labeled-sounds/) | **Original Repository:** [anuragkr90/webly-labeled-sounds](https://github.com/anuragkr90/webly-labeled-sounds)

**Description:**
This project implements methods for learning sound event recognition using web-sourced training data with imperfect labels. It enables machine learning models to effectively learn audio classification tasks from datasets gathered from online sources that may contain label noise.

**Key Features:**
- Audio dataset processing and feature extraction from YouTube sources
- Training/test split management for 40-dimensional embeddings
- Scripts for annotation and model evaluation
- Support for weakly-supervised learning from imperfect web labels

**Technical Stack:**
- **Languages:** Python (86.9%), Shell (13.1%)
- **Focus:** Audio processing, machine learning for sound event detection

**Project Structure:**
- `annotation/` - Annotation tools and data
- `features/` - Feature extraction modules
- `lists/` - Dataset file lists
- `scripts/` - Processing scripts
- `runscripts/` - Execution scripts

**Authors:** Anurag Kumar, Ankit Shah, Bhiksha Raj, Alexander Hauptmann

**Publication:** [Learning Sound Events from Webly Labeled Data](https://doi.org/10.24963/ijcai.2019/384) - IJCAI 2019

---

### 6. Learning from Weak Labels (Interspeech 2022 Tutorial + ICML 2024 Framework)

**Code:** [`learning_from_weak_labels/`](./learning_from_weak_labels/) | **Original Repositories:** [cmu-mlsp/learning_from_weak_labels](https://github.com/cmu-mlsp/learning_from_weak_labels), [Hhhhhhao/General-Framework-Weak-Supervision](https://github.com/Hhhhhhao/General-Framework-Weak-Supervision)

**Description:**
Comprehensive resources for learning from imprecise/weak labels, including the Interspeech 2022 tutorial materials and the ICML 2024 general framework implementation. Covers theoretical foundations, practical guidance, and state-of-the-art methods for weak label learning.

**Key Features:**
- Tutorial slides on weak label learning methodologies
- Reference implementations in MATLAB and Python
- **General Framework (ICML 2024):** Unified approach for 14+ weak supervision settings
  - Handles inexact (incomplete), incomplete (unlabeled), and inaccurate (corrupted) supervision
  - Models weak supervision as Non-deterministic Finite Automata (NFA)
  - Supports partial labels, noisy labels, semi-supervised, multiple instance learning, and more

**Technical Stack:**
- **Languages:** MATLAB, Python
- **Framework:** Based on USB (Universal Semi-supervised Learning) codebase
- **Training:** GPU-enabled (CUDA support)

**Project Structure:**
- `code/` - Reference implementations and examples (Interspeech tutorial)
- `General-Framework-Weak-Supervision/` - ICML 2024 unified framework implementation

**Presenters:** Bhiksha Raj, Anurag Kumar, Ankit Shah

**Publications:**
- [Interspeech 2022 Tutorial](https://www.interspeech2022.org/)
- [A General Framework for Learning from Weak Supervision](https://arxiv.org/abs/2402.01922) - ICML 2024
- [Imprecise Label Learning](https://arxiv.org/abs/2305.12715)

---