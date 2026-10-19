# Dataset Provenance, Attribution & License Notice

## 1. Dataset Overview

- **Dataset Title:** Student Performance Data Set
- **Repository:** UCI Machine Learning Repository
- **URL:** [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **DOI:** 10.24432/C5TG7T
- **Subject Area:** Social Sciences / Education / Learning Analytics

## 2. Citation & Authorship

Please cite the original paper when referencing or using this dataset in academic works:

> P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.

## 3. Data File Description

The project utilizes `student-mat.csv`, which documents secondary school students' achievement in mathematics across two Portuguese public schools (Gabriel Pereira and Mousinho da Silveira).

### Key Target:
- **`G3`**: Final year grade (numeric scale: 0 to 20).

### Key Features Used in Modeling:
- **`G1`**: First period grade (numeric: 0 to 20).
- **`G2`**: Second period grade (numeric: 0 to 20).
- **`studytime`**: Weekly study time (numeric ordinal: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, 4 - >10 hours).
- **`failures`**: Number of past class failures (numeric: n if 1 <= n < 3, else 4).
- **`absences`**: Number of school absences (numeric: 0 to 93).
- **`schoolsup`**: Extra educational school support (binary: 'yes' or 'no').
- **`famsup`**: Family educational support (binary: 'yes' or 'no').
- **`activities`**: Extra-curricular activities (binary: 'yes' or 'no').
- **`higher`**: Desires to take higher education (binary: 'yes' or 'no').
- **`internet`**: Internet access at home (binary: 'yes' or 'no').

## 4. Academic & Responsible Interpretation Policy

This project is an educational data mining prototype for learning analytics coursework.
- **Correlations do not prove causation:** High absences or low earlier grades correlate with lower final grades, but individual outcomes vary based on multiple social, personal, and pedagogical factors.
- **Advisory Prototype:** Predictions should only be interpreted as an early indicator to support educators in proactive mentoring and guidance.
- **No Consequential Automated Decisions:** Predictions must never be used to penalize, track, or restrict students' academic opportunities.
