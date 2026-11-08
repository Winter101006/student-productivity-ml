"""
Step 3: Unsupervised Clustering Pipeline (Section 2.3)
- Elbow Method & Silhouette Analysis (K=2 to 10) for K-Means
- BIC / AIC Analysis for Gaussian Mixture Models (GMM)
- DBSCAN density-based clustering with noise detection
- Quantitative Internal Metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Quantitative External Metrics: ARI, NMI, Homogeneity, Completeness, Purity
- Cross-tabulation between Clusters and True Labels
- Mandatory Dual-Encoding Scatter Plot: Color = Cluster, Marker = True Class Label
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def cluster_purity(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

def run_clustering():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_dir, 'data', 'processed', 'student_data_processed.csv')
    metrics_dir = os.path.join(base_dir, 'reports', 'metrics')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("[STEP 3] Loading dataset for clustering...")
    df = pd.read_csv(processed_path)
    feature_cols = [c for c in df.columns if c not in ['Productivity_Score', 'Productivity_Class']]
    X = df[feature_cols].values
    y_true = df['Productivity_Class'].values
    
    # 2D PCA representation for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca2 = pca_2d.fit_transform(X)
    
    # ==========================================
    # 1. Optimal K Search: Elbow & Silhouette for K-Means
    # ==========================================
    print("Evaluating K-Means for K = 2 to 10...")
    k_range = range(2, 11)
    inertias = []
    sil_scores_kmeans = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        sil_scores_kmeans.append(sil)
        
    df_kmeans_k = pd.DataFrame({
        'K': list(k_range),
        'Inertia_WCSS': inertias,
        'Silhouette_Score': sil_scores_kmeans
    })
    df_kmeans_k.to_csv(os.path.join(metrics_dir, '03_kmeans_elbow_silhouette.csv'), index=False, encoding='utf-8-sig')
    
    # Plot Elbow & Silhouette curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=7)
    ax1.set_title('Phương Pháp Elbow (Độ Lắng Quán Tính WCSS)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Số Lượng Cụm (K)', fontsize=10)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=10)
    ax1.set_xticks(k_range)
    
    ax2.plot(k_range, sil_scores_kmeans, 'rs--', linewidth=2, markersize=7)
    ax2.set_title('Hệ Số Silhouette Score Theo Số Lượng Cụm K', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Số Lượng Cụm (K)', fontsize=10)
    ax2.set_ylabel('Silhouette Score', fontsize=10)
    ax2.set_xticks(k_range)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '03_kmeans_elbow_silhouette.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 2. Optimal K Search for GMM: BIC & AIC
    # ==========================================
    print("Evaluating GMM for K = 2 to 10...")
    bics = []
    aics = []
    sil_scores_gmm = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        gmm_labels = gmm.predict(X)
        sil_scores_gmm.append(silhouette_score(X, gmm_labels))
        
    df_gmm_k = pd.DataFrame({
        'K': list(k_range),
        'BIC': bics,
        'AIC': aics,
        'Silhouette_Score': sil_scores_gmm
    })
    df_gmm_k.to_csv(os.path.join(metrics_dir, '03_gmm_bic_aic.csv'), index=False, encoding='utf-8-sig')
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, bics, 'o-', label='BIC (Bayesian Information Criterion)', color='#1d3557', linewidth=2)
    plt.plot(k_range, aics, 's--', label='AIC (Akaike Information Criterion)', color='#e76f51', linewidth=2)
    plt.title('Đánh Giá Tiêu Chí Thông Tin BIC & AIC Của Mô Hình GMM', fontsize=11, fontweight='bold')
    plt.xlabel('Số Lượng Thành Phần Hỗn Hợp (K)', fontsize=10)
    plt.ylabel('Giá Trị Tiêu Chuẩn (Thấp hơn là tốt hơn)', fontsize=10)
    plt.xticks(k_range)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '03_gmm_bic_aic.png'), dpi=300)
    plt.close()
    
    # ==========================================
    # 3. Fit 3 Clustering Algorithms (K=3 matches domain Low, Medium, High)
    # ==========================================
    print("Fitting K-Means, GMM, and DBSCAN...")
    k_opt = 3
    
    # (i) K-Means
    km_opt = KMeans(n_clusters=k_opt, random_state=42, n_init=20)
    kmeans_labels = km_opt.fit_predict(X)
    
    # (ii) GMM
    gmm_opt = GaussianMixture(n_components=k_opt, covariance_type='full', random_state=42)
    gmm_labels = gmm_opt.fit(X).predict(X)
    
    # (iii) DBSCAN
    # Calibrate eps for 35 dimensions standardized
    dbscan = DBSCAN(eps=4.2, min_samples=15)
    dbscan_labels = dbscan.fit_predict(X)
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN identified {n_dbscan_clusters} clusters and {(dbscan_labels == -1).sum()} noise points.")
    
    # ==========================================
    # 4. Evaluation Metrics (Internal & External)
    # ==========================================
    models = {
        'K-Means (K=3)': kmeans_labels,
        'GMM (K=3)': gmm_labels,
        f'DBSCAN (eps=4.2)': dbscan_labels
    }
    
    metrics_summary = []
    
    for name, labels in models.items():
        # Mask out noise for DBSCAN internal metrics if needed or compute on valid
        valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
        n_clusters_found = len(set(labels[valid_mask]))
        
        if n_clusters_found > 1:
            sil = silhouette_score(X[valid_mask], labels[valid_mask])
            db_idx = davies_bouldin_score(X[valid_mask], labels[valid_mask])
            ch_idx = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
        else:
            sil, db_idx, ch_idx = np.nan, np.nan, np.nan
            
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        homo = homogeneity_score(y_true, labels)
        comp = completeness_score(y_true, labels)
        v_meas = v_measure_score(y_true, labels)
        purity = cluster_purity(y_true[valid_mask], labels[valid_mask])
        
        metrics_summary.append({
            'Thuật Toán Phân Cụm': name,
            'Số Cụm Phát Hiện': n_clusters_found,
            'Silhouette Score (↑)': round(sil, 4) if not np.isnan(sil) else None,
            'Davies-Bouldin Index (↓)': round(db_idx, 4) if not np.isnan(db_idx) else None,
            'Calinski-Harabasz Index (↑)': round(ch_idx, 2) if not np.isnan(ch_idx) else None,
            'Adjusted Rand Index - ARI (↑)': round(ari, 4),
            'Normalized Mutual Info - NMI (↑)': round(nmi, 4),
            'Homogeneity (↑)': round(homo, 4),
            'Completeness (↑)': round(comp, 4),
            'V-Measure (↑)': round(v_meas, 4),
            'Purity (Độ Thuần Khiết) (↑)': round(purity, 4)
        })
        
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(metrics_dir, '03_clustering_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # Cross-tabulations
    ct_km = pd.crosstab(pd.Series(kmeans_labels, name='KMeans_Cluster'), pd.Series(y_true, name='True_Productivity_Class'))
    ct_km.to_csv(os.path.join(metrics_dir, '03_crosstab_kmeans.csv'), encoding='utf-8-sig')
    
    ct_gmm = pd.crosstab(pd.Series(gmm_labels, name='GMM_Cluster'), pd.Series(y_true, name='True_Productivity_Class'))
    ct_gmm.to_csv(os.path.join(metrics_dir, '03_crosstab_gmm.csv'), encoding='utf-8-sig')
    
    ct_db = pd.crosstab(pd.Series(dbscan_labels, name='DBSCAN_Cluster'), pd.Series(y_true, name='True_Productivity_Class'))
    ct_db.to_csv(os.path.join(metrics_dir, '03_crosstab_dbscan.csv'), encoding='utf-8-sig')
    
    # ==========================================
    # 5. Mandatory Dual-Encoding Scatter Plot: Color = Cluster, Marker = True Class
    # ==========================================
    print("Generating mandatory dual-encoding scatter plots...")
    marker_map = {'Low': 'o', 'Medium': 's', 'High': '^'} # Circle, Square, Triangle
    cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    plot_configs = [
        ('K-Means (K=3)', kmeans_labels, axes[0]),
        ('GMM (Expectation-Maximization K=3)', gmm_labels, axes[1]),
        ('DBSCAN (Mật Độ)', dbscan_labels, axes[2])
    ]
    
    for title, labels, ax in plot_configs:
        unique_lbls = sorted(list(set(labels)))
        for lbl in unique_lbls:
            color = '#7f7f7f' if lbl == -1 else cluster_colors[lbl % len(cluster_colors)]
            c_label = 'Noise (-1)' if lbl == -1 else f'Cụm {lbl}'
            
            for true_cls, marker in marker_map.items():
                mask = (labels == lbl) & (y_true == true_cls)
                if np.sum(mask) > 0:
                    ax.scatter(
                        X_pca2[mask, 0], X_pca2[mask, 1],
                        c=color, marker=marker,
                        s=28, alpha=0.6,
                        edgecolors='k', linewidth=0.3
                    )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Thành Phần Chính 1 (PC1)', fontsize=10)
        ax.set_ylabel('Thành Phần Chính 2 (PC2)', fontsize=10)
        
    # Legend construction
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333', markersize=8, label='Nhãn Thật: Thấp (Low) [Tròn]'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#333333', markersize=8, label='Nhãn Thật: Trung Bình (Medium) [Vuông]'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#333333', markersize=8, label='Nhãn Thật: Cao (High) [Tam Giác]'),
        Line2D([0], [0], color=cluster_colors[0], lw=4, label='Màu Sắc: Cụm 0'),
        Line2D([0], [0], color=cluster_colors[1], lw=4, label='Màu Sắc: Cụm 1'),
        Line2D([0], [0], color=cluster_colors[2], lw=4, label='Màu Sắc: Cụm 2'),
        Line2D([0], [0], color='#7f7f7f', lw=4, label='Màu Sắc: Điểm Nhiễu Noise (-1)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=10)
    plt.suptitle('Trực Quan Hóa Phân Cụm Trên Không Gian 2D PCA:\nMàu Sắc = Cụm Phân Bố (Cluster) | Hình Dạng Điểm = Nhãn Thực Tế (True Label)',
                 fontsize=14, fontweight='bold', y=1.12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '03_clustering_pca_dual_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[SUCCESS] Step 3 Clustering completed successfully.")

if __name__ == '__main__':
    run_clustering()
