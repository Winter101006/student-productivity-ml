# 🎓 Student Productivity and Digital Distraction Analysis
### End-to-End Machine Learning Pipeline: From Raw Digital Habits to Multi-Class Classification & Regression Transformation

[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67+-FF4B4B.svg)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Pipeline-100%25%20Verified%20%26%20Tested-brightgreen.svg)]()

---

## 📌 Project Overview
In modern higher education, digital devices (smartphones, laptops, tablets) are simultaneously indispensable study aids and the primary driver of **digital distraction**. Constant notifications, algorithmic social feeds, and prolonged screen exposure degrade sleep architecture, induce chronic cognitive fatigue, and directly depress Academic GPA.

This project delivers an **industrial-grade End-to-End Data Science & Machine Learning Pipeline**:
- Analyzes **3,500 student records** across **20 initial features** combining numerical and multi-type categorical data.
- Executes comprehensive **Data Preprocessing & Quality Engineering** (Missing data imputation, IQR outlier capping, Ordinal/One-Hot encoding, and standard scaling to 35 dimensions).
- Investigates dimensionality reduction comparing **PCA** vs **TruncatedSVD** (Scree plot, Kaiser-Guttman threshold, 2D/3D projections, and component loadings).
- Explores unsupervised cluster structure comparing **K-Means**, **Gaussian Mixture Models (EM)**, and **DBSCAN** across both internal and external clustering metrics.
- Evaluates a full experimental matrix of **7 classifiers** across **3 train-val splits** and **2 data representations**, diagnosing and mitigating **overfitting** via rigorous regularization.
- Mathematically formulates and validates a **continuous regression transformation** via classifier decision functions, contrasting full 35D space vs $\lfloor D/3 \rfloor = 11D$ space.
- Features an interactive **Streamlit Web Application** for exploratory data analytics, model comparisons, and real-time individualized student productivity forecasting.

---

## 🏗️ System Architecture Pipeline

```text
+-----------------------------------------------------------------------------------+
|                           1. RAW DATA INGESTION                                   |
|   3,500 Student Records | 20 Behavioral & Academic Features | Real Noise & Outliers|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                        2. DATA PREPROCESSING PIPELINE                             |
|  - Median/Mode Imputation   - IQR Outlier Capping    - Discretization (Tertiles)  |
|  - Ordinal Encoding         - OneHotEncoder (D: 20->35) - StandardScaler (Z-Score)|
+--------------------+------------------------------------+-------------------------+
                     |                                    |
                     v                                    v
+--------------------+----------------+  +----------------+-------------------------+
|     3. DIMENSIONALITY REDUCTION     |  |       4. UNSUPERVISED CLUSTERING         |
|  - PCA (90% Var = 23 Components)    |  |  - K-Means (Elbow & Silhouette Peak K=3) |
|  - TruncatedSVD Comparison          |  |  - GMM (EM - BIC/AIC Minimization)       |
|  - Scree Plot & Loadings Heatmap    |  |  - DBSCAN (Curse of Dimensionality)      |
|  - Pairwise Projections (PC1 - PC4) |  |  - Dual Scatter: Color=Cluster, Shape=Y  |
+--------------------+----------------+  +------------------------------------------+
                     |
                     v
+--------------------+--------------------------------------------------------------+
|                     5. SUPERVISED CLASSIFICATION BENCHMARK                        |
|  - Group 1: KNN, Decision Tree, Gaussian Naive Bayes (Generative)                 |
|  - Group 2: Multinomial Logistic Regression (Discriminative), MLP (Nonlinear)      |
|  - Group 3: Linear SVM, RBF SVM                                                   |
|  - 3 Splits: 80:20 (4:1), 70:30 (7:3), 60:40 (6:4) | 2 Data Spaces (35D vs 23D)   |
|  - Overfitting Diagnosis & Regularization (L2 Penalty, Tree Pruning, EarlyStop)   |
|  - Multiclass ROC Curves (AUC > 0.93), Precision-Recall & Normalized Confusion Mat |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                  6. REGRESSION TRANSFORMATION VIA DECISION SCORES                 |
|  - Target Extraction: y_reg = decision_function(X)[:, High]                       |
|  - Full Space (35D): Ridge Regression achieves R^2 = 1.0000 (Exact Recovery)      |
|  - Reduced Space (floor(D/3) = 11D PCA): 44.99% Variance Discarded -> R^2 = 0.9062|
|  - Mathematical Variance Decomposition & Residual Distribution Analysis           |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
|                 7. PRODUCTION PACKAGING & STREAMLIT DASHBOARD                     |
|  - Interactive Dashboard (app.py)       - Pre-computed Jupyter Notebook (.ipynb)  |
|  - High-Res Figures (reports/figures/)   - Comprehensive Academic Report (.md)     |
+-----------------------------------------------------------------------------------+
```

---

## 📊 Key Experimental Metrics & Findings

### 1. Classification & Regularization Impact (80:20 Split)
| Algorithm | Pre-Reg Train Acc | Pre-Reg Val Acc | Pre-Reg Gap | Post-Reg Train Acc | Post-Reg Val Acc | Post-Reg Gap | Gap Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | 1.0000 | 0.6757 | 0.3243 | 0.7186 | 0.6843 | **0.0343** | **+0.2900** |
| **MLP (Neural Net)** | 1.0000 | 0.7614 | 0.2386 | 0.8389 | 0.8143 | **0.0246** | **+0.2139** |
| **RBF SVM** | 1.0000 | 0.7614 | 0.2386 | 0.9086 | **0.8157** | **0.0929** | **+0.1457** |
| **Logistic Regression (L2)**| 0.8129 | 0.8071 | 0.0057 | 0.8139 | **0.8086** | **0.0054** | Optimal stability |
| **Gaussian Naive Bayes** | 0.7375 | 0.7729 | -0.0354 | 0.7375 | 0.7729 | -0.0354 | Generative baseline |

### 2. Clustering Performance (K=3)
| Clustering Method | Silhouette (↑) | Davies-Bouldin (↓) | Calinski-Harabasz (↑) | ARI (↑) | NMI (↑) | Purity (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means (K=3)** | **0.0580** | **3.4970** | **235.13** | **0.2276** | **0.2250** | **59.14%** |
| **GMM (EM, K=3)** | 0.0594 | 3.6162 | 149.68 | 0.0009 | 0.0016 | 35.57% |
| **DBSCAN (eps=4.2)** | 0.0026 | 2.0119 | 9.84 | 0.0036 | 0.0074 | 39.49% |

### 3. Regression Transformation (Target: Continuous High-Class Logit)
| Input Dimension Space | Regression Model | MSE (↓) | RMSE (↓) | MAE (↓) | $R^2$-Score (↑) | Theoretical Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Space ($D = 35$)** | **Ridge Regression** | **0.0000** | **0.0013** | **0.0011** | **1.0000** | Logit is exact linear combination $w^T x + b$ |
| Full Space ($D = 35$) | Random Forest | 2.2984 | 1.5160 | 1.2039 | 0.9026 | Orthogonal step approximation |
| **Reduced Space ($\lfloor D/3 \rfloor = 11$)** | **Ridge Regression** | **2.2134** | **1.4877** | **1.1656** | **0.9062** | 44.99% variance lost in 24 discarded axes |
| Reduced Space ($\lfloor D/3 \rfloor = 11$) | Random Forest | 2.8474 | 1.6874 | 1.3328 | 0.8793 | Information loss exacerbates tree error |

---

## 📁 Repository Structure

```text
project_root/
├── data/
│   ├── raw/
│   │   └── student_productivity_distraction.csv # 3,500 raw student records
│   └── processed/
│       ├── student_data_processed.csv           # 35D standardized dataset
│       └── student_data_pca.csv                 # 23D PCA transformed (90% variance)
├── src/
│   ├── data_generator.py                        # Dataset synthesis engine
│   ├── step1_preprocessing.py                   # Imputation, IQR Capping, OHE, Scaling
│   ├── step2_eda_dim_reduction.py               # EDA, Heatmap, PCA & TruncatedSVD
│   ├── step3_clustering.py                      # K-Means, GMM, DBSCAN & Dual-plot
│   ├── step4_classification_reg.py              # Matrix benchmark, Overfitting & Regression
│   └── build_notebook.py                        # Programmatic notebook generator
├── notebooks/
│   └── Full_ML_Pipeline_Report.ipynb            # Fully executed 14.7MB notebook with outputs
├── reports/
│   ├── figures/                                 # 17 Publication-quality PNGs (300 DPI)
│   ├── metrics/                                 # 18 Quantitative CSV metric tables
│   └── BAO_CAO_CHI_TIET.md                      # Comprehensive academic thesis report
├── app.py                                       # Production Streamlit Interactive Dashboard
├── requirements.txt                             # Frozen package dependencies
└── README.md                                    # Portfolio project documentation
```

---

## 🚀 Getting Started & How to Run

### 1. Environment Setup
```bash
# Clone or navigate to the repository directory
cd d:/Duan/ML_PLoai_SinhVien_SuDungThietBiSo

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Data Pipeline Steps
```bash
# Generate raw data (if needed)
python src/data_generator.py

# Step 1: Data Preprocessing & Scaling (Section 2.1)
python src/step1_preprocessing.py

# Step 2: EDA & Dimensionality Reduction (Section 2.2)
python src/step2_eda_dim_reduction.py

# Step 3: Unsupervised Clustering (Section 2.3)
python src/step3_clustering.py

# Step 4: Classification Benchmark & Regression Transformation (Section 2.4)
python src/step4_classification_reg.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
*Access the local web interface at `http://localhost:8501` to explore interactive 2D/3D PCA charts, cluster cross-tabs, confusion matrices, and the live student productivity predictor!*

### 4. Inspect or Re-run the Jupyter Notebook
```bash
jupyter notebook notebooks/Full_ML_Pipeline_Report.ipynb
```
*All 17 figures and benchmark tables are pre-executed and directly visible inline.*

---

## 👨‍💻 Key Skills Demonstrated for CV / Internship
- **Mathematical Rigor:** Eigendecomposition of covariance matrices, singular value decomposition, information criteria (BIC/AIC), cost-complexity pruning, and variance loss mechanics.
- **Data Engineering:** Missing data imputation, robust outlier capping via IQR, ordinal vs nominal dummy variable management, and zero-leakage pipeline design.
- **Applied Machine Learning:** Hyperparameter regularization (L2 weight decay, tree pruning, early stopping), multiclass ROC/PR analysis, and generative vs discriminative trade-offs.
- **Production Delivery:** Modular clean code, automated end-to-end execution, interactive dashboard deployment (Streamlit), and clear, academic documentation in Vietnamese and English.
