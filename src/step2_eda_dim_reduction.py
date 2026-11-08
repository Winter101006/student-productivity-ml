"""
Step 2: EDA & Dimensionality Reduction (Section 2.2)
- Correlation Heatmap, Skewness, Kurtosis
- Feature vs Target visualizations (Boxplot, Violin, KDE)
- PCA (Principal Component Analysis) & TruncatedSVD
- Eigenvalues, Explained Variance Ratio, Cumulative Explained Variance
- Principal Component Loadings Matrix
- Pairwise grid scatter plot for first 4-6 components
- 2D and 3D projection plots
- Method comparison metrics & export
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD

# Set consistent matplotlib styling for 300 DPI publication quality
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def run_eda_dim_reduction():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_dir, 'data', 'processed', 'student_data_processed.csv')
    metrics_dir = os.path.join(base_dir, 'reports', 'metrics')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("[STEP 2] Loading processed dataset for EDA & Dimensionality Reduction...")
    df = pd.read_csv(processed_path)
    
    # Feature columns vs Target
    feature_cols = [c for c in df.columns if c not in ['Productivity_Score', 'Productivity_Class']]
    X = df[feature_cols].values
    y = df['Productivity_Class'].values
    y_score = df['Productivity_Score'].values
    
    class_palette = {'Low': '#e63946', 'Medium': '#e9c46a', 'High': '#2a9d8f'}
    class_order = ['Low', 'Medium', 'High']
    
    # ==========================================
    # 1. Correlation Matrix & Heatmap
    # ==========================================
    print("Computing correlation matrix...")
    corr_matrix = df[feature_cols].corr()
    corr_matrix.to_csv(os.path.join(metrics_dir, '02_correlation_matrix.csv'), encoding='utf-8-sig')
    
    # Selected top 12 primary features for high-clarity heatmap
    top_features = [
        'Daily_Screen_Time_Hours', 'Study_Hours_Per_Day', 'Social_Media_Hours',
        'Gaming_Hours', 'Phone_Unlocks_Per_Day', 'Notification_Count_Daily',
        'Sleep_Duration_Hours', 'Physical_Activity_Minutes', 'Attendance_Rate_Percentage',
        'Current_GPA', 'Academic_Level_Ordinal', 'Stress_Level_Ordinal'
    ]
    top_corr = df[top_features].corr()
    
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(top_corr, dtype=bool))
    sns.heatmap(top_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Ma Trận Tương Quan Các Biến Số Chính (Pearson Correlation Heatmap)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 2. Relationship between Features and Target (Boxplot, Violin, KDE)
    # ==========================================
    print("Generating feature-target relationship charts...")
    # (a) Boxplots of key academic and digital features
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    box_features = [
        ('Current_GPA', 'Điểm GPA Tích Lũy (Thang 4.0)'),
        ('Study_Hours_Per_Day', 'Giờ Tự Học / Ngày'),
        ('Daily_Screen_Time_Hours', 'Giờ Sử Dụng Màn Hình / Ngày'),
        ('Social_Media_Hours', 'Giờ Dùng Mạng Xã Hội / Ngày'),
        ('Sleep_Duration_Hours', 'Thời Lượng Giấc Ngủ (Giờ)'),
        ('Phone_Unlocks_Per_Day', 'Số Lần Mở Khóa Điện Thoại / Ngày')
    ]
    for idx, (col, title) in enumerate(box_features):
        ax = axes[idx // 3, idx % 3]
        sns.boxplot(data=df, x='Productivity_Class', y=col, order=class_order,
                    palette=class_palette, ax=ax, width=0.5, fliersize=2)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Phân Lớp Hiệu Suất (Productivity Class)', fontsize=10)
        ax.set_ylabel('Giá Trị Chuẩn Hóa (Z-Score)', fontsize=10)
    plt.suptitle('Phân Phối Các Thuộc Tính Then Chốt Theo Nhãn Phân Loại (Boxplots)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_feature_boxplots_by_target.png'), dpi=300)
    plt.close()
    
    # (b) Violin Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    violin_cols = [
        ('Current_GPA', 'Phân Phối Mật Độ GPA'),
        ('Daily_Screen_Time_Hours', 'Mật Độ Thời Gian Màn Hình'),
        ('Study_Hours_Per_Day', 'Mật Độ Thời Gian Tự Học')
    ]
    for idx, (col, title) in enumerate(violin_cols):
        ax = axes[idx]
        sns.violinplot(data=df, x='Productivity_Class', y=col, order=class_order,
                       palette=class_palette, ax=ax, inner='quartile', cut=0)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Phân Lớp Hiệu Suất', fontsize=10)
        ax.set_ylabel('Giá Trị Chuẩn Hóa', fontsize=10)
    plt.suptitle('Biểu Đồ Mật Độ Violin Của Các Biến Đầu Vào Phân Tách Theo Lớp', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_feature_violin_plots.png'), dpi=300)
    plt.close()
    
    # (c) KDE Distribution of Continuous Productivity Score
    plt.figure(figsize=(9, 5))
    for cls in class_order:
        subset = df[df['Productivity_Class'] == cls]
        sns.kdeplot(subset['Productivity_Score'], label=f'Class: {cls}', fill=True, alpha=0.35, color=class_palette[cls], linewidth=2)
    plt.title('Hàm Mật Độ Xác Suất (KDE) Điểm Hiệu Suất Theo Phân Lớp', fontsize=12, fontweight='bold')
    plt.xlabel('Điểm Hiệu Suất Gốc (Productivity Score)', fontsize=10)
    plt.ylabel('Mật Độ Xác Suất (Density)', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_feature_kde_distribution.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 3. Dimensionality Reduction: PCA & TruncatedSVD
    # ==========================================
    print("Executing Dimensionality Reduction (PCA & TruncatedSVD)...")
    n_features = X.shape[1]
    
    # (A) PCA
    pca_full = PCA(random_state=42)
    X_pca_full = pca_full.fit_transform(X)
    pca_eigenvalues = pca_full.explained_variance_
    pca_exp_var_ratio = pca_full.explained_variance_ratio_
    pca_cum_var = np.cumsum(pca_exp_var_ratio)
    
    # (B) TruncatedSVD
    svd_full = TruncatedSVD(n_components=n_features, random_state=42)
    X_svd_full = svd_full.fit_transform(X)
    svd_exp_var_ratio = svd_full.explained_variance_ratio_
    svd_cum_var = np.cumsum(svd_exp_var_ratio)
    
    # Scree Plot & Cumulative Variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scree plot (Eigenvalues)
    ax1.plot(range(1, 16), pca_eigenvalues[:15], 'o-', color='#1d3557', linewidth=2, label='PCA Trị Riêng (Eigenvalue $\\lambda_i$)')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label=r'Ngưỡng Kaiser-Guttman ($\lambda=1$)')
    ax1.set_title('Biểu Đồ Scree Plot - Trị Riêng Của Từng Thành Phần', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Thứ Tự Thành Phần Chính (Principal Component)', fontsize=10)
    ax1.set_ylabel('Trị Riêng (Eigenvalue)', fontsize=10)
    ax1.set_xticks(range(1, 16))
    ax1.legend()
    
    # Cumulative Explained Variance
    ax2.plot(range(1, n_features + 1), pca_cum_var * 100, 's-', color='#2a9d8f', linewidth=2, label='PCA Phương Sai Tích Lũy (%)')
    ax2.plot(range(1, n_features + 1), svd_cum_var * 100, '^--', color='#e76f51', linewidth=1.8, label='TruncatedSVD Phương Sai Tích Lũy (%)')
    ax2.axhline(y=80, color='gray', linestyle=':', label='Ngưỡng 80%')
    ax2.axhline(y=90, color='black', linestyle=':', label='Ngưỡng 90%')
    ax2.set_title('Phương Sai Giải Thích Tích Lũy (Cumulative Explained Variance)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Số Lượng Chiều Giữ Lại (Dimensions)', fontsize=10)
    ax2.set_ylabel('Phương Sai Giải Thích Tích Lũy (%)', fontsize=10)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_scree_plot_pca_svd.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 4. Loadings Matrix (Trọng số biến gốc lên Principal Components)
    # ==========================================
    print("Computing Component Loadings...")
    # Loadings = Components * sqrt(Eigenvalues)
    pca_loadings = pca_full.components_[:6, :].T * np.sqrt(pca_eigenvalues[:6])
    df_loadings = pd.DataFrame(
        pca_loadings,
        index=feature_cols,
        columns=[f'PC{i+1}' for i in range(6)]
    )
    df_loadings.to_csv(os.path.join(metrics_dir, '02_pca_loadings.csv'), encoding='utf-8-sig')
    
    # Heatmap of top 12 variable loadings on first 4 PCs
    plt.figure(figsize=(9, 8))
    top_loadings = df_loadings.loc[top_features, ['PC1', 'PC2', 'PC3', 'PC4']]
    sns.heatmap(top_loadings, annot=True, fmt='.2f', cmap='vlag', center=0, cbar_kws={"shrink": 0.8})
    plt.title('Trọng Số Loadings Của Biến Gốc Lên 4 Thành Phần Chính Đầu Tiên', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_pca_loadings_heatmap.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 5. Pairwise Grid Scatter Plots for 4-6 Components
    # ==========================================
    print("Visualizing pairwise component grid (PC1 to PC4)...")
    pca_6 = PCA(n_components=6, random_state=42)
    X_pca_6 = pca_6.fit_transform(X)
    df_pca_6 = pd.DataFrame(X_pca_6, columns=[f'PC{i+1}' for i in range(6)])
    df_pca_6['Productivity_Class'] = df['Productivity_Class']
    
    # Pairwise plots of PC1, PC2, PC3, PC4 (6 pairs)
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, (p1, p2) in enumerate(pairs):
        ax = axes[idx // 3, idx % 3]
        for cls in class_order:
            sub = df_pca_6[df_pca_6['Productivity_Class'] == cls]
            ax.scatter(sub[f'PC{p1+1}'], sub[f'PC{p2+1}'], c=class_palette[cls],
                       label=cls, alpha=0.45, edgecolors='none', s=18)
        ax.set_title(f'PC{p1+1} vs PC{p2+1}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'PC{p1+1} ({pca_exp_var_ratio[p1]*100:.1f}%)', fontsize=9)
        ax.set_ylabel(f'PC{p2+1} ({pca_exp_var_ratio[p2]*100:.1f}%)', fontsize=9)
        if idx == 0:
            ax.legend(title='Productivity')
    plt.suptitle('Trực Quan Hóa Từng Cặp Thành Phần Chính (Pairwise PCA Projection Grid)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_pca_pairwise_grid.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 6. 2D and 3D Scatter Visualizations
    # ==========================================
    # 2D PCA Scatter Plot
    plt.figure(figsize=(9, 7))
    for cls in class_order:
        sub = df_pca_6[df_pca_6['Productivity_Class'] == cls]
        plt.scatter(sub['PC1'], sub['PC2'], c=class_palette[cls], label=cls, alpha=0.55, edgecolors='w', s=35, linewidth=0.3)
    plt.title(f'Không Gian Phân Bố 2D PCA (PC1 vs PC2) Phân Màu Theo Nhãn Hiệu Suất\n(Giải thích { (pca_exp_var_ratio[0]+pca_exp_var_ratio[1])*100:.1f}% Phương Sai)',
              fontsize=12, fontweight='bold')
    plt.xlabel(f'PC1 ({pca_exp_var_ratio[0]*100:.1f}% variance)', fontsize=11)
    plt.ylabel(f'PC2 ({pca_exp_var_ratio[1]*100:.1f}% variance)', fontsize=11)
    plt.legend(title='Productivity Class')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_pca_2d_scatter.png'), dpi=300)
    plt.close()
    
    # 3D PCA Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cls in class_order:
        sub = df_pca_6[df_pca_6['Productivity_Class'] == cls]
        ax.scatter(sub['PC1'], sub['PC2'], sub['PC3'], c=class_palette[cls], label=cls, alpha=0.55, s=25)
    ax.set_title('Không Gian 3D PCA (PC1 - PC2 - PC3)', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca_exp_var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_exp_var_ratio[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca_exp_var_ratio[2]*100:.1f}%)')
    ax.legend(title='Productivity Class')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_pca_3d_scatter.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 7. Comparison Table: PCA vs TruncatedSVD
    # ==========================================
    # Find components needed for variance thresholds
    def find_n_comp(cum_var, threshold):
        return int(np.argmax(cum_var >= threshold) + 1)
        
    comp_metrics = [
        {"Tiêu Chí Đánh Giá": "Số thành phần đạt >= 50% phương sai", "PCA": find_n_comp(pca_cum_var, 0.50), "TruncatedSVD": find_n_comp(svd_cum_var, 0.50)},
        {"Tiêu Chí Đánh Giá": "Số thành phần đạt >= 75% phương sai", "PCA": find_n_comp(pca_cum_var, 0.75), "TruncatedSVD": find_n_comp(svd_cum_var, 0.75)},
        {"Tiêu Chí Đánh Giá": "Số thành phần đạt >= 85% phương sai", "PCA": find_n_comp(pca_cum_var, 0.85), "TruncatedSVD": find_n_comp(svd_cum_var, 0.85)},
        {"Tiêu Chí Đánh Giá": "Số thành phần đạt >= 90% phương sai", "PCA": find_n_comp(pca_cum_var, 0.90), "TruncatedSVD": find_n_comp(svd_cum_var, 0.90)},
        {"Tiêu Chí Đánh Giá": "Số thành phần đạt >= 95% phương sai", "PCA": find_n_comp(pca_cum_var, 0.95), "TruncatedSVD": find_n_comp(svd_cum_var, 0.95)},
        {"Tiêu Chí Đánh Giá": "Tổng phương sai 2 chiều đầu tiên (PC1+PC2)", "PCA": f"{(pca_cum_var[1]*100):.2f}%", "TruncatedSVD": f"{(svd_cum_var[1]*100):.2f}%"},
        {"Tiêu Chí Đánh Giá": "Tổng phương sai 5 chiều đầu tiên", "PCA": f"{(pca_cum_var[4]*100):.2f}%", "TruncatedSVD": f"{(svd_cum_var[4]*100):.2f}%"},
        {"Tiêu Chí Đánh Giá": "Xử lý dữ liệu định tâm (Centering)", "PCA": "Bắt buộc trừ kỳ vọng (Zero-mean centering)", "TruncatedSVD": "Không bắt buộc định tâm (Phù hợp cả ma trận thưa)"},
        {"Tiêu Chí Đánh Giá": "Bản chất toán học", "PCA": r"Phân rã trị riêng ma trận hiệp phương sai $\Sigma$", "TruncatedSVD": r"Phân rã giá trị suy biến trực tiếp $X = U\Sigma V^T$"},
        {"Tiêu Chí Đánh Giá": "Ứng dụng tối ưu", "PCA": "Dữ liệu dạng bảng liên tục đã chuẩn hóa", "TruncatedSVD": "Dữ liệu ma trận thưa (Sparse text, TF-IDF, Graph)"}
    ]
    df_comparison = pd.DataFrame(comp_metrics)
    df_comparison.to_csv(os.path.join(metrics_dir, '02_dim_reduction_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # Save PCA transformed dataset with optimal components (retaining ~90% variance)
    n_opt_90 = find_n_comp(pca_cum_var, 0.90)
    pca_opt = PCA(n_components=n_opt_90, random_state=42)
    X_pca_opt = pca_opt.fit_transform(X)
    pca_cols = [f'PCA_Dim_{i+1}' for i in range(n_opt_90)]
    df_pca_opt = pd.DataFrame(X_pca_opt, columns=pca_cols)
    df_pca_opt['Productivity_Score'] = df['Productivity_Score']
    df_pca_opt['Productivity_Class'] = df['Productivity_Class']
    df_pca_opt.to_csv(os.path.join(base_dir, 'data', 'processed', 'student_data_pca.csv'), index=False, encoding='utf-8-sig')
    
    print(f"[SUCCESS] Step 2 EDA & Dim Reduction complete. PCA 90% variance uses {n_opt_90} components.")

if __name__ == '__main__':
    run_eda_dim_reduction()
