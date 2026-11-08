"""
Step 4: Classification, Overfitting Diagnosis, Regularization & Regression Transformation (Section 2.4)
- 3 Classifier Groups:
  * Group 1: KNN, Decision Tree, Gaussian Naive Bayes (Generative)
  * Group 2: Multinomial Logistic Regression (Linear Discriminative), MLP (Nonlinear)
  * Group 3: Linear SVM, RBF SVM
- 3 Train:Validation splits: 80:20 (4:1), 70:30 (7:3), 60:40 (6:4)
- 2 Data versions: Original Preprocessed (35D) vs PCA Reduced (90% variance)
- Overfitting diagnosis & Regularization comparison (Before vs After)
- Confusion Matrix (Counts & Normalized), Multiclass ROC Curves & PR Curves
- Part (c): Regression transformation via Decision Function (Class High)
  * Full D dimensions vs floor(D/3) dimensions via PCA
  * Ridge Regression & Random Forest Regressor
  * MSE, RMSE, MAE, R2-Score & Residual plots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def run_classification_and_regression():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_dir, 'data', 'processed', 'student_data_processed.csv')
    pca_path = os.path.join(base_dir, 'data', 'processed', 'student_data_pca.csv')
    metrics_dir = os.path.join(base_dir, 'reports', 'metrics')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("[STEP 4] Loading data for Classification & Regression experiments...")
    df_orig = pd.read_csv(processed_path)
    df_pca = pd.read_csv(pca_path)
    
    feat_orig = [c for c in df_orig.columns if c not in ['Productivity_Score', 'Productivity_Class']]
    feat_pca = [c for c in df_pca.columns if c not in ['Productivity_Score', 'Productivity_Class']]
    
    X_orig = np.array(df_orig[feat_orig].values, dtype=np.float64)
    X_pca = np.array(df_pca[feat_pca].values, dtype=np.float64)
    y = np.array(df_orig['Productivity_Class'].astype(str))
    
    classes = ['Low', 'Medium', 'High']
    split_ratios = [
        ('4:1 (80:20)', 0.20),
        ('7:3 (70:30)', 0.30),
        ('6:4 (60:40)', 0.40)
    ]
    
    # ==========================================
    # 1. Classification Matrix Across 3 Splits & 2 Data Versions
    # ==========================================
    print("Running classification experiment matrix...")
    
    def get_models(regularized=False):
        if not regularized:
            return {
                # Group 1:
                'KNN (k=1, Unregularized)': KNeighborsClassifier(n_neighbors=1),
                'Decision Tree (Max Depth None)': DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=42),
                'Gaussian Naive Bayes (Generative)': GaussianNB(),
                # Group 2:
                'Logistic Regression (C=100)': LogisticRegression(C=100.0, max_iter=1000, random_state=42),
                'MLP (No L2, EarlyStop=False)': MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.0, early_stopping=False, max_iter=300, random_state=42),
                # Group 3:
                'Linear SVM (C=10)': SVC(kernel='linear', C=10.0, probability=True, random_state=42),
                'RBF SVM (C=50, Gamma=Scale)': SVC(kernel='rbf', C=50.0, gamma='scale', probability=True, random_state=42)
            }
        else:
            return {
                # Group 1:
                'KNN (k=9, Regularized)': KNeighborsClassifier(n_neighbors=9, weights='distance'),
                'Decision Tree (Pruned, depth=5)': DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, ccp_alpha=0.005, random_state=42),
                'Gaussian Naive Bayes (Generative)': GaussianNB(var_smoothing=1e-8),
                # Group 2:
                'Logistic Regression (L2, C=0.5)': LogisticRegression(C=0.5, penalty='l2', max_iter=1000, random_state=42),
                'MLP (L2=0.01, EarlyStop=True)': MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, early_stopping=True, max_iter=300, random_state=42),
                # Group 3:
                'Linear SVM (C=0.5)': SVC(kernel='linear', C=0.5, probability=True, random_state=42),
                'RBF SVM (C=1.0, Regularized)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            }
            
    matrix_results = []
    
    for split_name, test_size in split_ratios:
        # Train-val split on Original
        X_tr_orig, X_val_orig, y_tr, y_val = train_test_split(
            X_orig, y, test_size=test_size, stratify=y, random_state=42
        )
        # Train-val split on PCA (same indices)
        X_tr_pca, X_val_pca, _, _ = train_test_split(
            X_pca, y, test_size=test_size, stratify=y, random_state=42
        )
        
        data_configs = [
            ('Dữ liệu gốc (35D)', X_tr_orig, X_val_orig),
            ('Dữ liệu PCA (23D - 90% Var)', X_tr_pca, X_val_pca)
        ]
        
        for data_label, X_tr, X_val in data_configs:
            models = get_models(regularized=False)
            for m_name, model in models.items():
                model.fit(X_tr, y_tr)
                y_tr_pred = model.predict(X_tr)
                y_val_pred = model.predict(X_val)
                
                acc_tr = accuracy_score(y_tr, y_tr_pred)
                acc_val = accuracy_score(y_val, y_val_pred)
                prec_macro = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
                rec_macro = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
                gap = acc_tr - acc_val
                
                matrix_results.append({
                    'Tỷ Lệ Phân Chia': split_name,
                    'Phiên Bản Dữ Liệu': data_label,
                    'Mô Hình': m_name,
                    'Train Accuracy': round(acc_tr, 4),
                    'Val Accuracy': round(acc_val, 4),
                    'Train-Val Gap (Overfit)': round(gap, 4),
                    'Precision (Macro)': round(prec_macro, 4),
                    'Recall (Macro)': round(rec_macro, 4),
                    'F1-Score (Macro)': round(f1_macro, 4),
                    'F1-Score (Weighted)': round(f1_weighted, 4)
                })
                
    df_matrix = pd.DataFrame(matrix_results)
    df_matrix.to_csv(os.path.join(metrics_dir, '04_classification_matrix_results.csv'), index=False, encoding='utf-8-sig')
    print(f"Computed {len(df_matrix)} experimental evaluation configurations.")
    
    # ==========================================
    # 2. Overfitting & Regularization Comparison (Before vs After)
    # ==========================================
    print("Evaluating Regularization Impact (Before vs After)...")
    # Using 80:20 split on Original data
    X_tr_orig, X_val_orig, y_tr, y_val = train_test_split(
        X_orig, y, test_size=0.20, stratify=y, random_state=42
    )
    
    unreg_models = get_models(regularized=False)
    reg_models = get_models(regularized=True)
    
    reg_comparison = []
    
    for (m_unreg_name, m_unreg), (m_reg_name, m_reg) in zip(unreg_models.items(), reg_models.items()):
        # Fit unregularized
        m_unreg.fit(X_tr_orig, y_tr)
        u_tr_acc = accuracy_score(y_tr, m_unreg.predict(X_tr_orig))
        u_val_acc = accuracy_score(y_val, m_unreg.predict(X_val_orig))
        u_gap = u_tr_acc - u_val_acc
        
        # Fit regularized
        m_reg.fit(X_tr_orig, y_tr)
        r_tr_acc = accuracy_score(y_tr, m_reg.predict(X_tr_orig))
        r_val_acc = accuracy_score(y_val, m_reg.predict(X_val_orig))
        r_gap = r_tr_acc - r_val_acc
        
        reg_comparison.append({
            'Thuật Toán Cơ Sở': m_unreg_name.split(' (')[0],
            'Trước Regularization (Train Acc)': round(u_tr_acc, 4),
            'Trước Regularization (Val Acc)': round(u_val_acc, 4),
            'Trước Gap (Overfit Độ Lệch)': round(u_gap, 4),
            'Sau Regularization (Train Acc)': round(r_tr_acc, 4),
            'Sau Regularization (Val Acc)': round(r_val_acc, 4),
            'Sau Gap (Overfit Độ Lệch)': round(r_gap, 4),
            'Giảm Overfit Gap': round(u_gap - r_gap, 4)
        })
        
    df_reg = pd.DataFrame(reg_comparison)
    df_reg.to_csv(os.path.join(metrics_dir, '04_regularization_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # Overfitting Bar Chart (Train vs Val before and after)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    model_labels = [r['Thuật Toán Cơ Sở'] for r in reg_comparison]
    x_pos = np.arange(len(model_labels))
    width = 0.35
    
    # Before
    u_tr = [r['Trước Regularization (Train Acc)'] for r in reg_comparison]
    u_val = [r['Trước Regularization (Val Acc)'] for r in reg_comparison]
    axes[0].bar(x_pos - width/2, u_tr, width, label='Train Accuracy', color='#e63946', alpha=0.85)
    axes[0].bar(x_pos + width/2, u_val, width, label='Val Accuracy', color='#457b9d', alpha=0.85)
    axes[0].set_title('Trước Hiệu Chỉnh (Chưa Regularization) - Overfit Cao', fontsize=11, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_labels, rotation=35, ha='right', fontsize=9)
    axes[0].set_ylabel('Độ Chính Xác (Accuracy)', fontsize=10)
    axes[0].legend()
    axes[0].set_ylim(0.4, 1.05)
    
    # After
    r_tr = [r['Sau Regularization (Train Acc)'] for r in reg_comparison]
    r_val = [r['Sau Regularization (Val Acc)'] for r in reg_comparison]
    axes[1].bar(x_pos - width/2, r_tr, width, label='Train Accuracy', color='#2a9d8f', alpha=0.85)
    axes[1].bar(x_pos + width/2, r_val, width, label='Val Accuracy', color='#e9c46a', alpha=0.85)
    axes[1].set_title('Sau Hiệu Chỉnh (Đã Regularization) - Gap Thu Hẹp Vững Chắc', fontsize=11, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_labels, rotation=35, ha='right', fontsize=9)
    axes[1].set_ylabel('Độ Chính Xác (Accuracy)', fontsize=10)
    axes[1].legend()
    axes[1].set_ylim(0.4, 1.05)
    
    plt.suptitle('So Sánh Hiện Tượng Overfitting & Hiệu Quả Của Regularization', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_overfitting_regularization_comparison.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 3. Part (b): Confusion Matrix & Multiclass ROC/PR Curves
    # ==========================================
    print("Generating Confusion Matrix & Multiclass ROC/PR curves for Best Model...")
    # Best model: Regularized Logistic Regression or RBF SVM
    best_clf = LogisticRegression(C=0.5, penalty='l2', max_iter=1000, random_state=42)
    best_clf.fit(X_tr_orig, y_tr)
    y_val_pred = best_clf.predict(X_val_orig)
    y_val_prob = best_clf.predict_proba(X_val_orig)
    
    # Confusion Matrix
    cm_counts = confusion_matrix(y_val, y_val_pred, labels=classes)
    cm_norm = confusion_matrix(y_val, y_val_pred, labels=classes, normalize='true')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Ma Trận Nhầm Lẫn (Số Lượng Mẫu - Absolute Counts)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Nhãn Dự Đoán (Predicted Label)', fontsize=10)
    ax1.set_ylabel('Nhãn Thực Tế (True Label)', fontsize=10)
    
    sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title('Ma Trận Nhầm Lẫn (Tỷ Lệ Chuẩn Hóa Theo Hàng %)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Nhãn Dự Đoán (Predicted Label)', fontsize=10)
    ax2.set_ylabel('Nhãn Thực Tế (True Label)', fontsize=10)
    
    plt.suptitle('Đánh Giá Ma Trận Nhầm Lẫn Của Mô Hình Tốt Nhất (Logistic Regression L2)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_confusion_matrices.png'), dpi=300)
    plt.close()
    
    # Multiclass ROC & Precision-Recall Curves (One-vs-Rest)
    y_val_bin = label_binarize(y_val, classes=classes)
    n_classes = len(classes)
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['#e63946', '#e9c46a', '#2a9d8f']
    
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_val_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[i], lw=2, label=f'Lớp {cls} (AUC = {roc_auc:.3f})')
        
        prec, rec, _ = precision_recall_curve(y_val_bin[:, i], y_val_prob[:, i])
        ap = average_precision_score(y_val_bin[:, i], y_val_prob[:, i])
        ax_pr.plot(rec, prec, color=colors[i], lw=2, label=f'Lớp {cls} (AP = {ap:.3f})')
        
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax_roc.set_title('Đường Cong Multiclass ROC (One-vs-Rest)', fontsize=11, fontweight='bold')
    ax_roc.set_xlabel('Tỷ Lệ Dương Tính Giả (False Positive Rate)', fontsize=10)
    ax_roc.set_ylabel('Tỷ Lệ Dương Tính Thật (True Positive Rate)', fontsize=10)
    ax_roc.legend(loc='lower right')
    
    ax_pr.set_title('Đường Cong Precision - Recall Cho Từng Lớp', fontsize=11, fontweight='bold')
    ax_pr.set_xlabel('Recall (Độ Thu Nhận)', fontsize=10)
    ax_pr.set_ylabel('Precision (Độ Chuẩn Xác)', fontsize=10)
    ax_pr.legend(loc='lower left')
    
    plt.suptitle('Đánh Giá Đường Cong ROC & Precision-Recall Cho Phân Loại 3 Lớp', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_roc_pr_curves.png'), dpi=300)
    plt.close()
    
    # Detailed Misclassification Diagnosis
    misclass_indices = np.where(y_val != y_val_pred)[0]
    misclass_records = []
    for idx in misclass_indices[:50]: # Sample 50 misclassified cases
        misclass_records.append({
            'Index_In_Val': int(idx),
            'True_Class': y_val[idx],
            'Predicted_Class': y_val_pred[idx],
            'Prob_Low': round(float(y_val_prob[idx, 0]), 3),
            'Prob_Medium': round(float(y_val_prob[idx, 1]), 3),
            'Prob_High': round(float(y_val_prob[idx, 2]), 3),
        })
    df_misclass = pd.DataFrame(misclass_records)
    df_misclass.to_csv(os.path.join(metrics_dir, '04_misclassification_analysis.csv'), index=False, encoding='utf-8-sig')
    
    # ==========================================
    # 4. Part (c): Regression Transformation via Decision Function
    # ==========================================
    print("Executing Part (c): Regression Transformation via Decision Function...")
    # Extract decision function score for Class 'High'
    # Decision function of Logistic Regression outputs logits (w^T x + b) for each class
    decision_scores = best_clf.decision_function(X_orig) # shape (N, 3)
    # Target class 'High' is index 2 (Low: 0, Medium: 1, High: 2)
    high_class_idx = list(best_clf.classes_).index('High')
    y_reg = decision_scores[:, high_class_idx]
    
    # Feature configurations:
    # 1. Full original input (D dimensions, D = 35)
    # 2. Reduced to floor(D / 3) dimensions via PCA
    D_full = X_orig.shape[1]
    D_reduced = int(np.floor(D_full / 3)) # 35 // 3 = 11
    print(f"Regression feature dimensions: Full D = {D_full} vs Reduced floor(D/3) = {D_reduced}")
    
    pca_reg = PCA(n_components=D_reduced, random_state=42)
    X_reg_pca13 = pca_reg.fit_transform(X_orig)
    pca13_var = np.sum(pca_reg.explained_variance_ratio_) * 100
    print(f"PCA with {D_reduced} components explains {pca13_var:.2f}% of total variance.")
    
    reg_models_dict = {
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }
    
    reg_results = []
    
    # Store predictions for visualization on 80:20 split
    viz_cache = {}
    
    for split_name, test_size in split_ratios:
        indices = np.arange(len(y_reg))
        tr_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        y_reg_tr, y_reg_val = y_reg[tr_idx], y_reg[val_idx]
        
        dim_cases = [
            (f'Dữ Liệu Đầy Đủ ({D_full} Chiều)', X_orig[tr_idx], X_orig[val_idx]),
            (f'Dữ Liệu Giảm Chiều 1/3 ({D_reduced} Chiều - {pca13_var:.1f}% Var)', X_reg_pca13[tr_idx], X_reg_pca13[val_idx])
        ]
        
        for case_name, X_tr_r, X_val_r in dim_cases:
            for reg_name, reg_model in reg_models_dict.items():
                reg_model.fit(X_tr_r, y_reg_tr)
                y_pred_val = reg_model.predict(X_val_r)
                
                mse = mean_squared_error(y_reg_val, y_pred_val)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_reg_val, y_pred_val)
                r2 = r2_score(y_reg_val, y_pred_val)
                
                reg_results.append({
                    'Tỷ Lệ Chia': split_name,
                    'Không Gian Đầu Vào': case_name,
                    'Mô Hình Hồi Quy': reg_name,
                    'MSE': round(mse, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'R2-Score': round(r2, 4)
                })
                
                if split_name == '4:1 (80:20)':
                    viz_cache[(case_name, reg_name)] = (y_reg_val, y_pred_val)
                    
    df_reg_results = pd.DataFrame(reg_results)
    df_reg_results.to_csv(os.path.join(metrics_dir, '04_regression_transformation_results.csv'), index=False, encoding='utf-8-sig')
    
    # ==========================================
    # 5. Regression Visualizations: Predicted vs Actual & Residuals
    # ==========================================
    # (a) Predicted vs Actual Scatter Plot (Full D vs D/3 for Ridge Regression)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    y_act_full, y_pred_full = viz_cache[(f'Dữ Liệu Đầy Đủ ({D_full} Chiều)', 'Ridge Regression')]
    y_act_red, y_pred_red = viz_cache[(f'Dữ Liệu Giảm Chiều 1/3 ({D_reduced} Chiều - {pca13_var:.1f}% Var)', 'Ridge Regression')]
    
    ax1.scatter(y_act_full, y_pred_full, alpha=0.45, color='#1d3557', edgecolors='none', s=25)
    lims1 = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]), max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
    ax1.plot(lims1, lims1, 'r--', lw=2, label='Đường Hoàn Hảo (y = x)')
    ax1.set_title(f'Không Gian Đầy Đủ ({D_full} Chiều) - Ridge Regression\n($R^2 = {r2_score(y_act_full, y_pred_full):.4f}$)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Điểm Quyết Định Thực Tế (Actual Decision Score)', fontsize=10)
    ax1.set_ylabel('Điểm Dự Đoán (Predicted Score)', fontsize=10)
    ax1.legend()
    
    ax2.scatter(y_act_red, y_pred_red, alpha=0.45, color='#e76f51', edgecolors='none', s=25)
    lims2 = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]), max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
    ax2.plot(lims2, lims2, 'r--', lw=2, label='Đường Hoàn Hảo (y = x)')
    ax2.set_title(f'Không Gian Giảm Chiều $\\lfloor D/3 \\rfloor$ ({D_reduced} Chiều) - Ridge Regression\n($R^2 = {r2_score(y_act_red, y_pred_red):.4f}$)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Điểm Quyết Định Thực Tế (Actual Decision Score)', fontsize=10)
    ax2.set_ylabel('Điểm Dự Đoán (Predicted Score)', fontsize=10)
    ax2.legend()
    
    plt.suptitle('So Sánh Tương Quan Giá Trị Dự Đoán vs Điểm Hàm Quyết Định Thực Tế', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_regression_pred_vs_actual.png'), dpi=300)
    plt.close()
    
    # (b) Residual Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    res_full = y_act_full - y_pred_full
    res_red = y_act_red - y_pred_red
    
    sns.histplot(res_full, kde=True, color='#1d3557', ax=ax1, stat='density', bins=30)
    ax1.axvline(0, color='red', linestyle='--', lw=1.5)
    ax1.set_title(f'Phân Phối Phần Dư ({D_full} Chiều)\nMSE: {mean_squared_error(y_act_full, y_pred_full):.4f} | Std: {np.std(res_full):.4f}', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Phần Dư (Residual: $y_{actual} - y_{pred}$)', fontsize=10)
    
    sns.histplot(res_red, kde=True, color='#e76f51', ax=ax2, stat='density', bins=30)
    ax2.axvline(0, color='red', linestyle='--', lw=1.5)
    ax2.set_title(f'Phân Phối Phần Dư ({D_reduced} Chiều PCA)\nMSE: {mean_squared_error(y_act_red, y_pred_red):.4f} | Std: {np.std(res_red):.4f}', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Phần Dư (Residual: $y_{actual} - y_{pred}$)', fontsize=10)
    
    plt.suptitle('Biểu Đồ Phân Phối Phần Dư (Residual Distributions) Giữa 2 Không Gian', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_regression_residuals.png'), dpi=300)
    plt.close()
    
    print("[SUCCESS] Step 4 Classification and Regression Transformation complete.")

if __name__ == '__main__':
    run_classification_and_regression()
