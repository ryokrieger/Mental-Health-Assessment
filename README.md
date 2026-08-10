# Mental Health Assessment Using Machine Learning (MHA)

A machine learning pipeline that classifies the overall **Mental Health Status (MHS)** of Bangladeshi university students into three categories — **Stable, Challenged, Critical** — from responses to three validated psychometric scales, and deploys the resulting model as a lightweight, stateless web assessment tool.

---

## Overview

University students face academic, social, and financial pressures that place them at elevated risk of stress, anxiety, and depression. Most existing screening work treats these three conditions separately, using a single psychometric scale at a time. This project instead builds a **combined, holistic mental-health classifier** that reasons jointly over three scales at once, and evaluates a broad range of feature selection strategies, classical ML models, deep learning architectures, and transformer-based NLP approaches to find the best-performing and most interpretable pipeline — before packaging the result into a live, self-assessment web app.

---

## Problem Statement

Existing machine-learning depression/anxiety-screening studies on Bangladeshi student populations share three recurring limitations:

1. They typically model **one psychometric scale in isolation** (e.g., only PHQ-9 or only a combined depression scale), rather than jointly reasoning over stress, anxiety, and depression as a single mental-health outcome.
2. Sample sizes are modest and drawn from **one or two institutions**, limiting generalisability.
3. Explainability, where present, is usually limited to a single technique (typically LIME only), and formal statistical comparison between competing models, calibration analysis, and the effect of demographic features are rarely reported.

This project addresses all three points using a larger, multi-institution dataset, three jointly modelled scales, a systematic feature-selection and model-optimisation sweep, and a dual explainability (SHAP + LIME) and statistical-testing evaluation layer.

---

## Dataset

The underlying data comes from a publicly available, IRB-approved survey of Bangladeshi university students (see [Data Source](#data-source--related-work) below), covering three academically adapted psychometric instruments:

| Scale | Measures | Items | Clinical range |
|---|---|---|---|
| **PSS-10** | Perceived stress | 10 (PSS1–PSS10) | 0–40 |
| **GAD-7** | Anxiety | 7 (GAD1–GAD7) | 0–21 |
| **PHQ-9** | Depression / mood | 9 (PHQ1–PHQ9) | 0–27 |

Each item was rephrased by the original survey authors to an academic-context equivalent (e.g., *"how often you felt as if you were unable to control important things in your academic affairs?"*), and responses were collected alongside seven demographic fields (age, gender, university, department, academic year, CGPA, scholarship status).

**Processed dataset used in this project:**

| Property | Value |
|---|---|
| Raw survey responses | 2,028 |
| Processed dataset (after duplicate removal) | **2,022 students** |
| Total columns | 40 (7 demographics + 26 scale items + 3 scale totals + 3 per-scale severity levels + 1 combined MHS label) |
| Universities represented | 15 (9 public, 6 private) |
| Class distribution | Critical 63.9% · Challenged 30.0% · Stable 6.1% |
| Class imbalance handling | SMOTE (applied to the training split only) |

### Deriving the combined Mental Health Status (MHS) label

The three-class MHS label is not part of the original survey — it is engineered in this project's preprocessing stage:

1. Each of PSS-10, GAD-7, and PHQ-9 is scored and independently classified into its own three-band severity level (Stable / Challenged / Critical) using standard clinical cut-off scores.
2. The three resulting per-scale labels (`Stress Level`, `Anxiety Level`, `Depression Level`) are combined by **majority vote**.
3. Because three voters can produce a genuine three-way split (one Stable, one Challenged, one Critical), the **PHQ-9 (depression) label is used as the deciding tiebreaker** in that case.

Two parallel datasets are produced from this pipeline:
- **`data/processed/mha_tabular_dataset.csv`** — the structured, numeric feature table used for all classical ML/DL/optimisation experiments.
- **`data/processed/mha_text_dataset.csv`** — a natural-language paragraph per student (demographics + all 26 scale responses translated into plain-English narrative), used for transformer/NLP fine-tuning.

---

## Methodology

The pipeline is organised as a sequence of Jupyter notebooks, each handling one stage of the workflow:

| Notebook | Stage | Summary |
|---|---|---|
| `01_Data_Preprocessing` | Cleaning & labelling | Column standardisation, reverse-scoring of positively worded PSS items, scale scoring, clinical thresholding, majority-vote MHS labelling, duplicate removal, EDA figures, text-narrative generation |
| `02_Feature_Engineering` | Feature selection | Nine independent feature-selection methods, each reducing the 26 scale items to a 15-feature subset |
| `03_Model_Training` | Baseline ML | Eight classical ML algorithms trained on each of the nine feature sets, plus CNN/ANN deep learning baselines |
| `04_Hyperparameter_Optimisation` | Tuning + DL | Grid Search, Randomised Search, and Bayesian optimisation of the top models; CNN and ANN deep-learning baselines |
| `05_NLP_Fine_Tuning` | Transformers | Fine-tuning of BERT, DistilBERT, BioBERT, ClinicalBERT, and ALBERT on the generated text narratives |
| `06_Explainability_Analysis` | Interpretability | Global SHAP (TreeExplainer, multi-class) and per-class LIME explanations for the best model |
| `07_Evaluation_Experiments` | Extended evaluation | ROC-AUC and calibration curves, statistical significance testing (McNemar's, Wilcoxon), a demographics-inclusion experiment, an ablation study, and a multi-output (per-scale) prediction experiment |

### Feature selection

Nine feature-selection methods — Recursive Feature Elimination (RFE), SelectKBest (SKB), Fisher Score / Chi-squared (FSCS), ExtraTreesClassifier importance (ETC), Pearson Correlation (PC), Mutual Information (MI), Mutual Information Regression (MIR), Manual Uniqueness (MU), and Variance Threshold (VT) — were each used to reduce the 26 scale items to a 15-item subset. Two pairs of methods converged on **identical** feature sets: SKB ≡ PC, and MI ≡ MIR. The two resulting sets differ by exactly one item (PSS2 vs. GAD2), a clean cross-validation of two independent statistical criteria.

### Modelling

- **Classical ML**: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, XGBoost, and LightGBM (eight algorithms), each trained across all nine feature sets.
- **Hyperparameter optimisation**: Grid Search, Randomised Search, and Bayesian optimisation applied to the strongest model/feature-set combinations.
- **Deep learning**: CNN and ANN architectures trained on the same tabular feature sets for comparison against classical ML.
- **NLP**: Five BERT-family transformers fine-tuned on the generated text narratives, to test whether a language-model approach to the same underlying information outperforms structured tabular modelling.

### Explainability & evaluation

- **SHAP** (TreeExplainer) for global, multi-class feature importance across the full test set.
- **LIME** for local, per-instance, per-class explanations.
- **ROC-AUC** (one-vs-rest) and **calibration curves** for each class.
- **McNemar's test** and **Wilcoxon signed-rank test** to formally compare the top two optimised models.
- A **demographics-inclusion experiment** testing whether adding the seven demographic fields to the 15 scale-item features improves or harms performance.
- An **ablation study** isolating the individual contribution of feature selection, SMOTE, and hyperparameter tuning.
- A **multi-output experiment** predicting stress, anxiety, and depression sub-scores simultaneously, to identify which construct is hardest to predict from the reduced item set.

---

## Key Results

- The best-performing model overall is a **Bayesian-optimised XGBoost classifier trained on the SKB (equivalently PC) feature set** — 15 items spanning all three scales — achieving **Macro F1 = 0.8667**, **Accuracy = 90.86%**, and **Macro ROC-AUC = 0.9792** on the held-out test set.
- Classical machine learning on structured tabular features **substantially outperformed** both deep learning (best CNN Macro F1 = 0.8595) and transformer-based NLP on the generated text narratives (best model, ClinicalBERT, Macro F1 = 0.6384).
- **SHAP** analysis identifies feeling down/hopeless (PHQ2), being easily annoyed/irritated (GAD4), and feeling like a failure (PHQ6) as the strongest global predictors of mental health status; PSS (stress) items consistently rank lowest in global importance, and stress is also the weakest-predicted construct in the multi-output experiment.
- Adding demographic features (age, gender, university, department, year, CGPA, scholarship) to the scale-item feature set **reduced** performance, with the damage concentrated almost entirely in the minority *Stable* class — evidence that demographics act mainly as noise for this task rather than as useful signal.
- The two top optimised models (Bayesian XGB+SKB and Bayesian LightGBM+FSCS) are **not statistically significantly different** at α = 0.05 (McNemar's and Wilcoxon tests), despite the former's marginally higher held-out score.
- Calibration curves are notably non-monotonic for all three classes given the limited test-set size (405 samples, only 25 in the minority Stable class) — the model's class *ranking* is strong, but raw predicted probabilities should not be over-interpreted as calibrated confidence, particularly for the Stable class.

## Deployed Web Application

A separate, stateless web app (`app/`) presents the 15 selected questions one at a time and returns a predicted mental health status with class probabilities and a written interpretation. There is no login and no database — each assessment is a single stateless request.

Because the deployment platform (Vercel/AWS Lambda) enforces a package-size limit that excludes both XGBoost and LightGBM, the **deployed model is not the best research model**. Among the deployable candidates, a **Randomised Search-optimised Gradient Boosting classifier trained on the MI feature set** was selected — tied for the best Macro F1 (0.8601) and with the best accuracy (91.11%) among the eligible tied group. The MI feature set differs from the research-best SKB set by exactly one item (GAD2 in place of PSS2), so the deployed app's 15 questions are correspondingly slightly different from the research model's feature list.

---

## Novel Contributions

1. **Joint, multi-scale MHS classification** — a combined Stable/Challenged/Critical label derived from PSS-10, GAD-7, and PHQ-9 together via majority vote (PHQ-9 tiebreak), rather than modelling any one scale in isolation.
2. **Larger, multi-institution sample** — 2,022 processed responses from 15 universities, roughly 3–4× larger than comparable prior Bangladeshi studies, which are typically drawn from a single institution.
3. **Systematic nine-method feature selection comparison**, revealing that two independent statistical criterion pairs converge on nearly identical 15-item subsets.
4. **Empirical evidence that demographic features act as noise** for this task, with effects concentrated in the minority class.
5. **Dual explainability** — global SHAP alongside per-instance LIME, rather than either technique alone.
6. **Formal statistical significance testing** (McNemar's, Wilcoxon) between competing top models.
7. **A multi-output sub-scale experiment** isolating a structural difficulty in predicting perceived stress relative to anxiety or depression.
8. **Calibration analysis** reported alongside ROC-AUC, surfacing a limitation (unreliable probability calibration for the minority class) not typically discussed in comparable prior work.

---

## Data Source

- **Dataset**: Syeed MM, Rahman A, Akter L, Fatema K, Khan RH, Karim MR, Hossain MS, Uddin MF. *A comprehensive standardized dataset on Mental Health Problems (MHPs) of University Students.* Public dataset: https://doi.org/10.6084/m9.figshare.25771164.v1

---

## Ethics & Data Privacy

The underlying survey was collected anonymously under an approved ethics protocol (IRB#SETS-2023-0409, Independent University, Bangladesh), following informed consent and the Declaration of Helsinki. This project performs secondary analysis on that already-anonymised, publicly released dataset; no personally identifiable information is collected, stored, or reproduced anywhere in this repository or the deployed web application, which itself is stateless and retains no user responses after a prediction is returned.

---

## Tech Stack

- **Data & ML**: NumPy, pandas, SciPy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, LightGBM, scikit-optimize
- **Deep learning**: PyTorch
- **NLP**: Hugging Face Transformers, Datasets, Tokenizers, Accelerate
- **Explainability**: SHAP, LIME
- **Statistical testing**: Pingouin, statsmodels
- **Deployment**: Vercel (static front end + Python serverless function)