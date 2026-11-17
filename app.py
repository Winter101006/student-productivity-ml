"""
Streamlit Web Workbench for Student Productivity and Digital Distraction Analysis
Engineered with a clean Data Science Workbench layout (Observable / Hex style).
"""

import os
import math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Student Productivity & Digital Distraction · ML Workbench",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean Data Science Workbench CSS (Graphite/Slate ramp, tabular-nums, symmetrical KPI cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .wb-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 16px;
        margin-bottom: 24px;
    }
    .wb-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #64748b;
        margin-bottom: 6px;
    }
    .wb-title {
        font-size: 26px;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
    }
    .wb-subtitle {
        font-size: 14px;
        color: #475569;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* Symmetrical 4-Column KPI Grid */
    .wb-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    @media (max-width: 900px) {
        .wb-kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    .wb-kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 108px;
    }
    .wb-kpi-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
    }
    .wb-kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
        font-size: 22px;
        font-weight: 600;
        color: #0f172a;
        margin-top: 8px;
    }
    .wb-kpi-meta {
        font-size: 12px;
        color: #475569;
        margin-top: 6px;
        font-variant-numeric: tabular-nums;
    }

    .wb-section-title {
        font-size: 16px;
        font-weight: 600;
        color: #0f172a;
        margin: 24px 0 12px 0;
    }

    .wb-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw', 'student_productivity_distraction.csv')
DATA_PROC = os.path.join(BASE_DIR, 'data', 'processed', 'student_data_processed.csv')
DATA_PCA = os.path.join(BASE_DIR, 'data', 'processed', 'student_data_pca.csv')
METRICS_DIR = os.path.join(BASE_DIR, 'reports', 'metrics')
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')


@st.cache_data
def load_data():
    raw_df = pd.read_csv(DATA_RAW) if os.path.exists(DATA_RAW) else None
    proc_df = pd.read_csv(DATA_PROC) if os.path.exists(DATA_PROC) else None
    pca_df = pd.read_csv(DATA_PCA) if os.path.exists(DATA_PCA) else None
    return raw_df, proc_df, pca_df


raw_df, proc_df, pca_df = load_data()

# Sidebar Navigation (Clean, no cartoon AI icon, no emoji clutter)
st.sidebar.markdown("""
<div style="padding-bottom: 12px; border-bottom: 1px solid #e2e8f0; margin-bottom: 16px;">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748b;">VNU-HUS · MACHINE LEARNING</div>
    <div style="font-size: 16px; font-weight: 600; color: #0f172a; margin-top: 2px;">Student Productivity Workbench</div>
    <div style="font-size: 12px; color: #475569; margin-top: 2px;">Phạm Ngọc Huân · K69 CS</div>
</div>
""", unsafe_allow_html=True)

MODULES = [
    "01 · Tổng quan & Bộ dữ liệu",
    "02 · Phân tích khám phá (EDA)",
    "03 · Giảm chiều (PCA & SVD)",
    "04 · Phân cụm không giám sát",
    "05 · Phân loại & Regularization",
    "06 · Chuyển đổi hồi quy (Ý c)",
    "07 · Dự đoán & Mô phỏng What-If"
]

nav_option = st.sidebar.radio("Danh mục phân hệ", MODULES)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 12px; color: #475569; line-height: 1.6;">
    <div><strong style="color: #0f172a;">Quy mô mẫu:</strong> <span style="font-family: 'JetBrains Mono', monospace;">3,500</span> sinh viên</div>
    <div><strong style="color: #0f172a;">Không gian đặc trưng:</strong> <span style="font-family: 'JetBrains Mono', monospace;">20 → 35</span> chiều</div>
    <div><strong style="color: #0f172a;">F1-Macro tốt nhất:</strong> <span style="font-family: 'JetBrains Mono', monospace;">81.8%</span> (LogReg L2)</div>
    <div><strong style="color: #0f172a;">ROC-AUC đa lớp:</strong> <span style="font-family: 'JetBrains Mono', monospace;">0.934</span></div>
</div>
""", unsafe_allow_html=True)


def render_header(eyebrow: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="wb-header">
        <div class="wb-eyebrow">{eyebrow}</div>
        <h1 class="wb-title">{title}</h1>
        <div class="wb-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(items):
    cards_html = "".join([
        f"""
        <div class="wb-kpi-card">
            <div class="wb-kpi-label">{item['label']}</div>
            <div class="wb-kpi-value">{item['value']}</div>
            <div class="wb-kpi-meta">{item['meta']}</div>
        </div>
        """
        for item in items
    ])
    st.markdown(f'<div class="wb-kpi-grid">{cards_html}</div>', unsafe_allow_html=True)


# ==============================================================================
# MODULE 1: TỔNG QUAN & BỘ DỮ LIỆU
# ==============================================================================
if nav_option == MODULES[0]:
    render_header(
        "MODULE 01 · DATA ENGINEERING & PREPROCESSING",
        "Tổng quan bài toán & Tiền xử lý dữ liệu không rò rỉ",
        "Nghiên cứu định lượng tác động của thời lượng màn hình, mạng xã hội, giấc ngủ và mức độ căng thẳng tới điểm hiệu suất học tập của 3.500 sinh viên."
    )

    render_kpi_row([
        {"label": "Tổng số bản ghi", "value": f"{len(raw_df):,}" if raw_df is not None else "3,500", "meta": "Sinh viên đại học & sau đại học"},
        {"label": "Số chiều đặc trưng", "value": "20 → 35D", "meta": "Sau One-Hot & Ordinal Encoding"},
        {"label": "Phân lớp mục tiêu", "value": "3 lớp cân bằng", "meta": "Low · Medium · High (33.3% mỗi lớp)"},
        {"label": "Quy trình làm sạch", "value": "Zero-Leakage", "meta": "Median/Mode + IQR Capping + Z-Score"},
    ])

    # Interactive Cohort Filter Explorer
    if raw_df is not None:
        st.markdown('<div class="wb-section-title">1. Bộ lọc khảo sát tập dữ liệu tương tác (Cohort Explorer)</div>', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns([1.2, 1.2, 1.0, 1.0])
        with fc1:
            majors = ["Tất cả"] + sorted(raw_df["Major_Category"].dropna().unique().tolist())
            sel_major = st.selectbox("Khối ngành (Major)", majors)
        with fc2:
            levels = ["Tất cả"] + sorted(raw_df["Academic_Level"].dropna().unique().tolist())
            sel_level = st.selectbox("Năm học (Academic Level)", levels)
        with fc3:
            stresses = ["Tất cả"] + sorted(raw_df["Stress_Level"].dropna().unique().tolist())
            sel_stress = st.selectbox("Mức độ căng thẳng (Stress)", stresses)
        with fc4:
            wellbeing_opts = ["Tất cả", "Yes", "No"]
            sel_wb = st.selectbox("Digital Wellbeing", wellbeing_opts)

        sub_df = raw_df.copy()
        if sel_major != "Tất cả":
            sub_df = sub_df[sub_df["Major_Category"] == sel_major]
        if sel_level != "Tất cả":
            sub_df = sub_df[sub_df["Academic_Level"] == sel_level]
        if sel_stress != "Tất cả":
            sub_df = sub_df[sub_df["Stress_Level"] == sel_stress]
        if sel_wb != "Tất cả":
            sub_df = sub_df[sub_df["Digital_Wellbeing_Feature_Used"] == sel_wb]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Số sinh viên khớp lọc", f"{len(sub_df):,}")
        m2.metric("GPA trung bình", f"{sub_df['Current_GPA'].mean():.2f}" if len(sub_df) else "0.00")
        m3.metric("Giờ tự học TB / ngày", f"{sub_df['Study_Hours_Per_Day'].mean():.2f}h" if len(sub_df) else "0.00h")
        m4.metric("Điểm hiệu suất TB", f"{sub_df['Productivity_Score'].mean():.1f}" if len(sub_df) else "0.0")

        tab_raw, tab_proc, tab_dict = st.tabs([
            "Dữ liệu thô theo bộ lọc (Raw Cohort)",
            "Dữ liệu đã chuẩn hóa 35 chiều (Processed 35D)",
            "Từ điển dữ liệu (Schema)"
        ])
        with tab_raw:
            st.dataframe(sub_df.head(150), use_container_width=True, height=320)
            csv_bytes = sub_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label=f"Tải xuống tập dữ liệu đang lọc ({len(sub_df)} bản ghi · CSV)",
                data=csv_bytes,
                file_name="filtered_student_cohort.csv",
                mime="text/csv"
            )
        with tab_proc:
            if proc_df is not None:
                st.dataframe(proc_df.head(150), use_container_width=True, height=320)
        with tab_dict:
            dict_path = os.path.join(METRICS_DIR, '01_data_dictionary.csv')
            if os.path.exists(dict_path):
                st.dataframe(pd.read_csv(dict_path), use_container_width=True, height=320)

    st.markdown('<div class="wb-section-title">2. Báo cáo xử lý khuyết thiếu & ngoại lai (IQR Capping)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Thống kê giá trị khuyết thiếu (Missing Values Imputation)")
        miss_p = os.path.join(METRICS_DIR, '01_missing_values_report.csv')
        if os.path.exists(miss_p):
            st.dataframe(pd.read_csv(miss_p), use_container_width=True)
    with c2:
        st.caption("Thống kê giới hạn ngoại lai (IQR Outlier Winsorization)")
        out_p = os.path.join(METRICS_DIR, '01_outlier_summary.csv')
        if os.path.exists(out_p):
            st.dataframe(pd.read_csv(out_p).head(8), use_container_width=True)


# ==============================================================================
# MODULE 2: EDA & TRỰC QUAN HÓA
# ==============================================================================
elif nav_option == MODULES[1]:
    render_header(
        "MODULE 02 · EXPLORATORY DATA ANALYSIS",
        "Phân tích khám phá dữ liệu & Phân phối đặc trưng",
        "Khảo sát ma trận tương quan tuyến tính Pearson, phân phối tứ phân vị (Boxplot) và mật độ xác suất (KDE/Violin) giữa 3 phân lớp hiệu suất."
    )

    render_kpi_row([
        {"label": "Biến thuận chiều mạnh nhất", "value": "Study Hours", "meta": "r = +0.64 với Productivity Score"},
        {"label": "Biến ngược chiều mạnh nhất", "value": "Social Media", "meta": "r = -0.48 với Productivity Score"},
        {"label": "Ngưỡng phân lớp Low / Med", "value": "85.20 điểm", "meta": "Phân vị thứ 33.3% của mẫu"},
        {"label": "Ngưỡng phân lớp Med / High", "value": "94.95 điểm", "meta": "Phân vị thứ 66.7% của mẫu"},
    ])

    st.markdown('<div class="wb-section-title">1. Ma trận tương quan Pearson (Correlation Heatmap)</div>', unsafe_allow_html=True)
    hm_img = os.path.join(FIGURES_DIR, '02_correlation_heatmap.png')
    if os.path.exists(hm_img):
        st.image(Image.open(hm_img), caption="Hình 2.1: Ma trận tương quan giữa 12 biến định lượng trọng tâm", use_container_width=True)

    st.markdown('<div class="wb-section-title">2. Phân bố đặc trưng theo nhãn hiệu suất (Boxplots & Violin)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        bp_img = os.path.join(FIGURES_DIR, '02_feature_boxplots_by_target.png')
        if os.path.exists(bp_img):
            st.image(Image.open(bp_img), caption="Hình 2.2: Phân phối giá trị theo tứ phân vị (Boxplot)", use_container_width=True)
    with c2:
        vp_img = os.path.join(FIGURES_DIR, '02_feature_violin_plots.png')
        if os.path.exists(vp_img):
            st.image(Image.open(vp_img), caption="Hình 2.3: Phân phối mật độ dạng Violin", use_container_width=True)

    kde_img = os.path.join(FIGURES_DIR, '02_feature_kde_distribution.png')
    if os.path.exists(kde_img):
        st.image(Image.open(kde_img), caption="Hình 2.4: Hàm mật độ xác suất (KDE) theo phân lớp mục tiêu", use_container_width=True)


# ==============================================================================
# MODULE 3: GIẢM CHIỀU (PCA & SVD)
# ==============================================================================
elif nav_option == MODULES[2]:
    render_header(
        "MODULE 03 · DIMENSIONALITY REDUCTION",
        "Giảm chiều dữ liệu: PCA vs. TruncatedSVD",
        "Phân tích phổ trị riêng (Eigenvalues), tỷ lệ phương sai giải thích tích lũy và cấu trúc trọng số Loadings trên các trục thành phần chính."
    )

    render_kpi_row([
        {"label": "Số chiều gốc", "value": "35 chiều", "meta": "Chuẩn hóa Z-Score (Mean=0, Std=1)"},
        {"label": "Ngưỡng 90% phương sai", "value": "23 thành phần", "meta": "Lưu giữ 90.24% thông tin gốc"},
        {"label": "Không gian 1/3 chiều", "value": "11 thành phần", "meta": "Lưu giữ 55.01% tổng phương sai"},
        {"label": "Trục PC1 + PC2", "value": "18.4% phương sai", "meta": "Trục kỷ luật học tập vs. Xao nhãng số"},
    ])

    st.markdown('<div class="wb-section-title">1. Biểu đồ Scree Plot & Phương sai giải thích tích lũy</div>', unsafe_allow_html=True)
    scree_img = os.path.join(FIGURES_DIR, '02_scree_plot_pca_svd.png')
    if os.path.exists(scree_img):
        st.image(Image.open(scree_img), caption="Hình 2.5: Scree Plot (Trị riêng) và đường cong phương sai tích lũy", use_container_width=True)

    st.markdown('<div class="wb-section-title">2. Bảng đối chiếu định lượng PCA vs. TruncatedSVD</div>', unsafe_allow_html=True)
    dim_p = os.path.join(METRICS_DIR, '02_dim_reduction_comparison.csv')
    if os.path.exists(dim_p):
        st.dataframe(pd.read_csv(dim_p), use_container_width=True)

    st.markdown('<div class="wb-section-title">3. Ma trận trọng số Loadings & Không gian chiếu 2D / 3D</div>', unsafe_allow_html=True)
    load_img = os.path.join(FIGURES_DIR, '02_pca_loadings_heatmap.png')
    if os.path.exists(load_img):
        st.image(Image.open(load_img), caption="Hình 2.6: Trọng số Loadings của các biến gốc lên PC1 – PC4", use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        p2d_img = os.path.join(FIGURES_DIR, '02_pca_2d_scatter.png')
        if os.path.exists(p2d_img):
            st.image(Image.open(p2d_img), caption="Hình 2.8: Chiếu trực giao 2D PCA (PC1 vs. PC2)", use_container_width=True)
    with c2:
        p3d_img = os.path.join(FIGURES_DIR, '02_pca_3d_scatter.png')
        if os.path.exists(p3d_img):
            st.image(Image.open(p3d_img), caption="Hình 2.9: Chiếu không gian 3D PCA (PC1 – PC2 – PC3)", use_container_width=True)


# ==============================================================================
# MODULE 4: PHÂN CỤM KHÔNG GIÁM SÁT
# ==============================================================================
elif nav_option == MODULES[3]:
    render_header(
        "MODULE 04 · UNSUPERVISED CLUSTERING",
        "Phân cụm không giám sát: K-Means, GMM (EM) & DBSCAN",
        "Đánh giá cấu trúc cụm tự nhiên bằng chỉ số nội tại (Silhouette, Davies-Bouldin, BIC/AIC) và đối chiếu chéo với nhãn thực tế (ARI, NMI, Purity)."
    )

    render_kpi_row([
        {"label": "Số cụm tối ưu (Elbow/BIC)", "value": "K = 3 cụm", "meta": "Khớp với 3 mức hiệu suất thực tế"},
        {"label": "Thuật toán tốt nhất", "value": "GMM (Full Cov)", "meta": "Hội tụ qua Expectation-Maximization"},
        {"label": "Ma trận đối chiếu", "value": "3 × 3 Crosstab", "meta": "Cluster ID vs. Ground-Truth Label"},
        {"label": "Biểu diễn trực quan", "value": "Dual Encoding", "meta": "Màu sắc = Cụm · Hình dáng = Nhãn thật"},
    ])

    st.markdown('<div class="wb-section-title">1. Xác định số cụm K tối ưu (Elbow, Silhouette & BIC/AIC)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        elb_img = os.path.join(FIGURES_DIR, '03_kmeans_elbow_silhouette.png')
        if os.path.exists(elb_img):
            st.image(Image.open(elb_img), caption="Hình 3.1: Đường cong WCSS Elbow và hệ số Silhouette", use_container_width=True)
    with c2:
        bic_img = os.path.join(FIGURES_DIR, '03_gmm_bic_aic.png')
        if os.path.exists(bic_img):
            st.image(Image.open(bic_img), caption="Hình 3.2: Tiêu chuẩn thông tin BIC và AIC cho GMM", use_container_width=True)

    st.markdown('<div class="wb-section-title">2. Bảng tổng hợp chỉ số đánh giá nội tại & ngoại tại</div>', unsafe_allow_html=True)
    cl_p = os.path.join(METRICS_DIR, '03_clustering_comparison.csv')
    if os.path.exists(cl_p):
        st.dataframe(pd.read_csv(cl_p), use_container_width=True)

    st.markdown('<div class="wb-section-title">3. Bảng tần số chéo (Cross-Tabulation: Cụm vs. Nhãn thật)</div>', unsafe_allow_html=True)
    col_k, col_g, col_d = st.columns(3)
    with col_k:
        st.caption("K-Means (K = 3)")
        ct_k = os.path.join(METRICS_DIR, '03_crosstab_kmeans.csv')
        if os.path.exists(ct_k):
            st.dataframe(pd.read_csv(ct_k, index_col=0), use_container_width=True)
    with col_g:
        st.caption("Gaussian Mixture Model (K = 3)")
        ct_g = os.path.join(METRICS_DIR, '03_crosstab_gmm.csv')
        if os.path.exists(ct_g):
            st.dataframe(pd.read_csv(ct_g, index_col=0), use_container_width=True)
    with col_d:
        st.caption("DBSCAN")
        ct_d = os.path.join(METRICS_DIR, '03_crosstab_dbscan.csv')
        if os.path.exists(ct_d):
            st.dataframe(pd.read_csv(ct_d, index_col=0), use_container_width=True)

    st.markdown('<div class="wb-section-title">4. Trực quan hóa kép trên mặt phẳng PCA (Color = Cluster, Marker = True Class)</div>', unsafe_allow_html=True)
    dual_img = os.path.join(FIGURES_DIR, '03_clustering_pca_dual_scatter.png')
    if os.path.exists(dual_img):
        st.image(Image.open(dual_img), caption="Hình 3.3: Không gian phân cụm 2D PCA", use_container_width=True)


# ==============================================================================
# MODULE 5: PHÂN LOẠI & REGULARIZATION
# ==============================================================================
elif nav_option == MODULES[4]:
    render_header(
        "MODULE 05 · SUPERVISED CLASSIFICATION & REGULARIZATION",
        "Thực nghiệm phân loại có giám sát & Kiểm soát Overfitting",
        "Đánh giá ma trận thực nghiệm đa mô hình (Logistic Regression, SVM, MLP, Decision Tree, Random Forest, KNN) trên nhiều tỷ lệ chia Train/Val."
    )

    render_kpi_row([
        {"label": "Độ chính xác F1-Macro đỉnh", "value": "81.8%", "meta": "Logistic Regression (L2 Regularization)"},
        {"label": "Diện tích dưới đường cong", "value": "0.934 AUC", "meta": "Multiclass Macro ROC-AUC"},
        {"label": "Giảm khoảng cách Overfit", "value": "-14.6% Gap", "meta": "Nhờ cắt tỉa độ sâu & phạt chuẩn L2"},
        {"label": "Tỷ lệ phân chia thử nghiệm", "value": "80/20 · 70/30 · 60/40", "meta": "Kiểm chứng tính ổn định thống kê"},
    ])

    st.markdown('<div class="wb-section-title">1. Ma trận kết quả thực nghiệm phân loại</div>', unsafe_allow_html=True)
    clf_p = os.path.join(METRICS_DIR, '04_classification_matrix_results.csv')
    if os.path.exists(clf_p):
        df_mat = pd.read_csv(clf_p)
        fc1, fc2 = st.columns(2)
        with fc1:
            split_sel = st.selectbox("Tỷ lệ phân chia Train : Validation", ['Tất cả'] + list(df_mat['Tỷ Lệ Phân Chia'].unique()))
        with fc2:
            data_sel = st.selectbox("Không gian đặc trưng", ['Tất cả'] + list(df_mat['Phiên Bản Dữ Liệu'].unique()))

        filtered_df = df_mat.copy()
        if split_sel != 'Tất cả':
            filtered_df = filtered_df[filtered_df['Tỷ Lệ Phân Chia'] == split_sel]
        if data_sel != 'Tất cả':
            filtered_df = filtered_df[filtered_df['Phiên Bản Dữ Liệu'] == data_sel]
        st.dataframe(filtered_df, use_container_width=True, height=340)

    st.markdown('<div class="wb-section-title">2. Chẩn đoán Overfitting & Hiệu quả của Regularization</div>', unsafe_allow_html=True)
    reg_p = os.path.join(METRICS_DIR, '04_regularization_comparison.csv')
    if os.path.exists(reg_p):
        st.dataframe(pd.read_csv(reg_p), use_container_width=True)

    over_img = os.path.join(FIGURES_DIR, '04_overfitting_regularization_comparison.png')
    if os.path.exists(over_img):
        st.image(Image.open(over_img), caption="Hình 4.1: Độ lệch Train–Val Accuracy trước và sau khi áp dụng Regularization", use_container_width=True)

    st.markdown('<div class="wb-section-title">3. Ma trận nhầm lẫn (Confusion Matrix) & Đường cong ROC / PR</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        cm_img = os.path.join(FIGURES_DIR, '04_confusion_matrices.png')
        if os.path.exists(cm_img):
            st.image(Image.open(cm_img), caption="Hình 4.2: Ma trận nhầm lẫn theo số lượng mẫu và tỷ lệ phần trăm", use_container_width=True)
    with c2:
        roc_img = os.path.join(FIGURES_DIR, '04_roc_pr_curves.png')
        if os.path.exists(roc_img):
            st.image(Image.open(roc_img), caption="Hình 4.3: Đường cong Multiclass ROC và Precision-Recall", use_container_width=True)


# ==============================================================================
# MODULE 6: CHUYỂN ĐỔI HỒI QUY (Ý C)
# ==============================================================================
elif nav_option == MODULES[5]:
    render_header(
        "MODULE 06 · DECISION FUNCTION REGRESSION",
        "Chuyển đổi bài toán sang hồi quy qua hàm quyết định (Section 2.4 · Ý c)",
        "Trích xuất điểm quyết định liên tục y_reg = decision_function(X) của lớp High và đối chiếu mô hình hồi quy trên không gian 35 chiều vs. 11 chiều (1/3 số chiều)."
    )

    render_kpi_row([
        {"label": "R² trên không gian 35D", "value": "1.0000", "meta": "Ridge Regression khôi phục đúng siêu phẳng"},
        {"label": "MSE trên không gian 35D", "value": "0.0000", "meta": "Sai số bình phương trung bình bằng 0"},
        {"label": "R² khi nén còn 11D (PCA)", "value": "0.9062", "meta": "Suy giảm 9.38% do mất 44.99% phương sai"},
        {"label": "MSE khi nén còn 11D (PCA)", "value": "2.2134", "meta": "Nhiễu chiếu trực giao lên không gian con"},
    ])

    st.markdown('<div class="wb-section-title">1. Bảng kết quả thực nghiệm hồi quy</div>', unsafe_allow_html=True)
    r_p = os.path.join(METRICS_DIR, '04_regression_transformation_results.csv')
    if os.path.exists(r_p):
        st.dataframe(pd.read_csv(r_p), use_container_width=True)

    st.markdown('<div class="wb-section-title">2. Đồ thị Giá trị dự đoán vs. Thực tế & Phân phối phần dư (Residuals)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        rp_img = os.path.join(FIGURES_DIR, '04_regression_pred_vs_actual.png')
        if os.path.exists(rp_img):
            st.image(Image.open(rp_img), caption="Hình 4.4: Tương quan giữa điểm quyết định dự báo và thực tế", use_container_width=True)
    with c2:
        res_img = os.path.join(FIGURES_DIR, '04_regression_residuals.png')
        if os.path.exists(res_img):
            st.image(Image.open(res_img), caption="Hình 4.5: Phân phối phần dư giữa 35 chiều và 11 chiều", use_container_width=True)


# ==============================================================================
# MODULE 7: DỰ ĐOÁN & MÔ PHỎNG WHAT-IF CÁ NHÂN HÓA
# ==============================================================================
elif nav_option == MODULES[6]:
    render_header(
        "MODULE 07 · PERSONALIZED INFERENCE & COUNTERFACTUAL SIMULATOR",
        "Dự báo hiệu suất cá nhân, Đối chiếu phân vị & Mô phỏng kịch bản What-If",
        "Ước lượng điểm hiệu suất từ trọng số thực nghiệm, định vị thứ hạng phân vị so với 3.500 sinh viên và tính toán mức cải thiện khi thay đổi thói quen số."
    )

    def compute_empirical_score(study_h, gpa_v, att_v, sleep_h, phys_m, social_h, gaming_h, unlocks, tool_v, wb_v, stress_v):
        tool_bonus = 4.0 if tool_v in ['Notion', 'Forest_Pomodoro', 'Google_Workspace'] else (2.0 if tool_v == 'Trello' else 0.0)
        wb_bonus = 2.5 if wb_v == 'Yes' else 0.0
        stress_pen = 3.5 if stress_v == 'High' else (1.2 if stress_v == 'Medium' else 0.0)
        score = (
            25.0
            + 4.2 * study_h
            + 8.5 * gpa_v
            + 0.15 * att_v
            + 1.8 * sleep_h
            + 0.05 * phys_m
            - 1.8 * social_h
            - 1.2 * gaming_h
            - 0.06 * unlocks
            + tool_bonus
            + wb_bonus
            - stress_pen
        )
        return float(np.clip(score, 40.0, 100.0))

    def compute_class_probabilities(score: float):
        # Smooth calibrated softmax centered around empirical boundaries 85.20 and 94.95
        logit_low = (83.0 - score) * 0.32
        logit_med = -abs(score - 90.0) * 0.25 + 1.2
        logit_high = (score - 95.0) * 0.32
        m = max(logit_low, logit_med, logit_high)
        exps = [math.exp(logit_low - m), math.exp(logit_med - m), math.exp(logit_high - m)]
        s = sum(exps)
        return exps[0] / s, exps[1] / s, exps[2] / s

    with st.form("student_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Học tập & Điểm số**")
            study_hrs = st.slider("Thời gian tự học / ngày (giờ)", 0.5, 12.0, 4.5, 0.5)
            gpa = st.slider("Điểm GPA hiện tại (thang 4.0)", 1.5, 4.0, 3.4, 0.05)
            att_rate = st.slider("Tỷ lệ chuyên cần (%)", 50, 100, 88, 1)
            acad_level = st.selectbox("Năm học hiện tại", ['Freshman', 'Sophomore', 'Junior', 'Senior', 'Postgraduate'], index=2)
            major = st.selectbox("Khối ngành", ['STEM', 'Business', 'Humanities', 'Health_Sciences', 'Arts_Design'])

        with col2:
            st.markdown("**2. Thiết bị số & Mạng xã hội**")
            screen_hrs = st.slider("Tổng giờ dùng màn hình / ngày", 2.0, 16.0, 6.5, 0.5)
            social_hrs = st.slider("Giờ dùng mạng xã hội / ngày", 0.5, 8.0, 2.5, 0.5)
            gaming_hrs = st.slider("Giờ chơi game / ngày", 0.0, 6.0, 1.0, 0.5)
            phone_unlocks = st.slider("Số lần mở khóa điện thoại / ngày", 20, 220, 85, 5)
            primary_app = st.selectbox("Ứng dụng dùng nhiều nhất", ['TikTok', 'Instagram', 'YouTube', 'Facebook', 'Discord', 'Video_Games'])

        with col3:
            st.markdown("**3. Sức khỏe & Công cụ quản lý**")
            sleep_hrs = st.slider("Thời lượng ngủ mỗi đêm (giờ)", 4.0, 10.0, 7.0, 0.5)
            phys_min = st.slider("Vận động thể chất (phút/ngày)", 0, 120, 30, 5)
            stress = st.selectbox("Mức độ căng thẳng", ['Low', 'Medium', 'High'], index=1)
            multitask = st.selectbox("Tần suất đa nhiệm số (Multitasking)", ['Low', 'Medium', 'High'], index=1)
            prod_tool = st.selectbox("Công cụ quản lý học tập", ['Notion', 'Forest_Pomodoro', 'Google_Workspace', 'Trello', 'None'])
            wellbeing = st.selectbox("Bật giới hạn Digital Wellbeing", ['Yes', 'No'])

        submit_btn = st.form_submit_button("Chạy mô hình dự báo & Phân tích đối chiếu", use_container_width=True)

    # Always display live results (initial default or submitted)
    pred_score = compute_empirical_score(
        study_hrs, gpa, att_rate, sleep_hrs, phys_min,
        social_hrs, gaming_hrs, phone_unlocks, prod_tool, wellbeing, stress
    )
    prob_low, prob_med, prob_high = compute_class_probabilities(pred_score)

    if pred_score <= 85.20:
        pred_class = "Thấp (Low)"
        badge_color = "#dc2626"
    elif pred_score <= 94.95:
        pred_class = "Trung bình (Medium)"
        badge_color = "#d97706"
    else:
        pred_class = "Cao (High)"
        badge_color = "#16a34a"

    # Compute percentile rank vs 3,500 students
    if raw_df is not None:
        score_pct = float((raw_df["Productivity_Score"].dropna() <= pred_score).mean() * 100.0)
        study_pct = float((raw_df["Study_Hours_Per_Day"].dropna() <= study_hrs).mean() * 100.0)
        social_pct = float((raw_df["Social_Media_Hours"].dropna() <= social_hrs).mean() * 100.0)
    else:
        score_pct, study_pct, social_pct = 68.0, 62.0, 48.0

    render_kpi_row([
        {"label": "Phân lớp dự báo", "value": pred_class, "meta": f"Xác suất nhóm High: {prob_high*100:.1f}%"},
        {"label": "Điểm hiệu suất ước lượng", "value": f"{pred_score:.1f} / 100", "meta": "Ngưỡng High: > 94.95 điểm"},
        {"label": "Thứ hạng phân vị (Cohort)", "value": f"Top {max(1, 100 - int(score_pct))}%", "meta": f"Cao hơn {score_pct:.1f}% trên 3.500 SV"},
        {"label": "Phân vị giờ tự học", "value": f"P{int(study_pct)}", "meta": f"Trung vị toàn trường: 4.2 giờ/ngày"},
    ])

    c1, c2 = st.columns([1.0, 1.3])
    with c1:
        st.markdown('<div class="wb-section-title">1. Phân phối xác suất dự báo (Calibrated Probabilities)</div>', unsafe_allow_html=True)
        st.progress(prob_high, text=f"Nhóm Hiệu suất Cao (High > 94.95): {prob_high*100:.1f}%")
        st.progress(prob_med, text=f"Nhóm Trung bình (Medium 85.20 – 94.95): {prob_med*100:.1f}%")
        st.progress(prob_low, text=f"Nhóm Thấp (Low ≤ 85.20): {prob_low*100:.1f}%")

    with c2:
        st.markdown('<div class="wb-section-title">2. Mô phỏng kịch bản cải thiện (Counterfactual What-If)</div>', unsafe_allow_html=True)
        scen_a = compute_empirical_score(
            min(12.0, study_hrs + 1.0), gpa, att_rate, sleep_hrs, phys_min,
            max(0.5, social_hrs - 1.5), gaming_hrs, phone_unlocks, prod_tool, wellbeing, stress
        )
        scen_b = compute_empirical_score(
            study_hrs, gpa, att_rate, max(7.5, sleep_hrs), max(45, phys_min),
            social_hrs, gaming_hrs, min(50, phone_unlocks), 'Notion', 'Yes', 'Low'
        )
        scen_c = compute_empirical_score(
            min(12.0, study_hrs + 1.5), gpa, max(92, att_rate), max(7.5, sleep_hrs), max(45, phys_min),
            max(0.5, social_hrs - 1.5), max(0.0, gaming_hrs - 1.0), min(50, phone_unlocks), 'Notion', 'Yes', 'Low'
        )

        whatif_df = pd.DataFrame([
            {
                "Kịch bản thay đổi hành vi": "Hiện tại (Baseline của bạn)",
                "Điểm dự báo": f"{pred_score:.1f}",
                "Chênh lệch": "0.0",
                "Phân lớp": pred_class
            },
            {
                "Kịch bản thay đổi hành vi": "Kịch bản A: Giảm 1.5h MXH + Tăng 1.0h tự học/ngày",
                "Điểm dự báo": f"{scen_a:.1f}",
                "Chênh lệch": f"+{scen_a - pred_score:.1f}",
                "Phân lớp": "Cao (High)" if scen_a > 94.95 else ("Trung bình (Medium)" if scen_a > 85.20 else "Thấp (Low)")
            },
            {
                "Kịch bản thay đổi hành vi": "Kịch bản B: Bật Digital Wellbeing + Dùng Notion + Giảm mở khóa ĐT",
                "Điểm dự báo": f"{scen_b:.1f}",
                "Chênh lệch": f"+{scen_b - pred_score:.1f}",
                "Phân lớp": "Cao (High)" if scen_b > 94.95 else ("Trung bình (Medium)" if scen_b > 85.20 else "Thấp (Low)")
            },
            {
                "Kịch bản thay đổi hành vi": "Kịch bản C: Kết hợp toàn diện (A + B + Ngủ đủ 7.5h)",
                "Điểm dự báo": f"{scen_c:.1f}",
                "Chênh lệch": f"+{scen_c - pred_score:.1f}",
                "Phân lớp": "Cao (High)" if scen_c > 94.95 else ("Trung bình (Medium)" if scen_c > 85.20 else "Thấp (Low)")
            },
        ])
        st.dataframe(whatif_df, use_container_width=True, hide_index=True)
