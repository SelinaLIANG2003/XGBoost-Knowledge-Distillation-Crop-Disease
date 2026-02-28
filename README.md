# Tomato Leaf Disease Classification via Tree-Based Knowledge Distillation

## Overview
This repository contains the code and research thesis for my graduation project: a lightweight crop disease classification model. The project addresses the computational constraints of agricultural edge devices by implementing a custom Knowledge Distillation (KD) framework using XGBoost.

## Abstract
**Background:** The rapid aging of the agricultural workforce necessitates highly efficient smart farming systems. While deep learning approaches achieve high accuracy, their computational requirements limit deployment on resource-constrained edge devices. 

**Methodology:** Utilizing a 10-class tomato leaf image dataset, I engineered 768-dimensional log-transformed RGB frequency distributions from alpha-masked images. To overcome the structural limitations of decision trees, I designed a novel XGBoost-to-XGBoost distillation pipeline. Knowledge was transferred from a high-capacity Teacher model to a shallow Student model via temperature scaling, top-m pseudo-label expansion, and dynamic sample weighting.

**Results:** Under the optimal configuration, my distilled student model (Student-KD) achieved an accuracy of 80.07% and maintained a highly robust disease recall of 98.93%. The proposed approach accelerated inference speed by 10.1% (7.7x faster than the Teacher model) without compromising classification performance.

## Files
* `xgbKD.py`: The main XGBoost Knowledge Distillation experimental script.
* `卒業論文.pdf`: Full graduation thesis (Japanese).
