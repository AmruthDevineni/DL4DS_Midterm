# DL4DS_Midterm
DL4DS_Midterm

This repository contains my completed submission for the **DS542 Deep Learning for Data Science – Spring 2025 Midterm Challenge** at Boston University.

## 🧠 Overview

This challenge involves training and evaluating deep learning models on the CIFAR-100 dataset across three parts:

| Part | Description |
|------|-------------|
| **Part 1** | A simple custom CNN model trained from scratch (`starter_code.py`) |
| **Part 2** | A deeper model using `torchvision.models.ResNet18` (`part2_resnet_train.py`) |
| **Part 3** | Transfer learning with a pretrained WideResNet50_2 model (`part3.py`) using advanced techniques like AutoAugment, label smoothing, EMA, and cosine learning rate scheduling |

---

## 📁 Repository Structure

```
.
├── starter_code.py           # Part 1: SimpleCNN model (from scratch)
├── part2_resnet_train.py     # Part 2: ResNet18 model training
├── part3.py                  # Part 3: WideResNet50_2 transfer learning
├── eval_cifar100.py          # Evaluation on CIFAR-100 test set
├── eval_ood.py               # Evaluation on OOD data
├── requirements.txt          # Python dependencies
├── DL4DS_Midterm_Report.docx # Full report (submitted separately)
└── README.md                 # Project summary
```

