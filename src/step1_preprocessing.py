"""
Step 1: Data Preprocessing Pipeline (Section 2.1)
- Data dictionary and schema definition
- Summary statistics (numerical and categorical)
- Missing values diagnosis and median/mode imputation
- Outlier detection (IQR & Z-score) and capping
- Discretization of continuous target into 3 classes (Low, Medium, High)
- Ordinal encoding for ordered categorical features
- One-Hot encoding for nominal categorical features
- Feature scaling with StandardScaler
- Final processed data export & metric reporting
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

def run_preprocessing():
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'raw', 'student_productivity_distraction.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    metrics_dir = os.path.join(base_dir, 'reports', 'metrics')
    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print("[STEP 1] Reading raw dataset...")
    # 'None' in Productivity_Tools_Used should be treated as a valid category 'None', not NaN
    df_raw = pd.read_csv(raw_path, keep_default_na=False)
    # Replace empty strings with NaN
    df_raw = df_raw.replace(r'^\s*$', np.nan, regex=True)
    
    # Convert numerical columns to float/int
    num_cols_raw = [
        'Age', 'Daily_Screen_Time_Hours', 'Study_Hours_Per_Day', 'Social_Media_Hours',
        'Gaming_Hours', 'Phone_Unlocks_Per_Day', 'Notification_Count_Daily',
        'Sleep_Duration_Hours', 'Physical_Activity_Minutes', 'Attendance_Rate_Percentage',
        'Current_GPA', 'Productivity_Score'
    ]
    for c in num_cols_raw:
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')
        
    print(f"Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    
    # 1. Data Dictionary & Metadata
    data_dict = [
        {"Column": "Student_ID", "Type": "Identifier", "Meaning": "Mã định danh duy nhất của sinh viên"},
        {"Column": "Age", "Type": "Numerical (Discrete)", "Meaning": "Độ tuổi sinh viên (18 - 28 tuổi)"},
        {"Column": "Gender", "Type": "Categorical (Nominal)", "Meaning": "Giới tính (Female, Male, Non-Binary)"},
        {"Column": "Academic_Level", "Type": "Categorical (Ordinal)", "Meaning": "Năm học đại học / sau đại học"},
        {"Column": "Major_Category", "Type": "Categorical (Nominal)", "Meaning": "Khối ngành học (STEM, Business, Humanities,...)"},
        {"Column": "Daily_Screen_Time_Hours", "Type": "Numerical (Continuous)", "Meaning": "Tổng thời gian dùng màn hình số hàng ngày (giờ)"},
        {"Column": "Study_Hours_Per_Day", "Type": "Numerical (Continuous)", "Meaning": "Thời gian tự học trung bình mỗi ngày (giờ)"},
        {"Column": "Social_Media_Hours", "Type": "Numerical (Continuous)", "Meaning": "Thời gian dùng mạng xã hội (giờ/ngày)"},
        {"Column": "Gaming_Hours", "Type": "Numerical (Continuous)", "Meaning": "Thời gian chơi game (giờ/ngày)"},
        {"Column": "Phone_Unlocks_Per_Day", "Type": "Numerical (Discrete)", "Meaning": "Số lần mở khóa điện thoại trong ngày"},
        {"Column": "Notification_Count_Daily", "Type": "Numerical (Discrete)", "Meaning": "Số lượng thông báo điện thoại nhận mỗi ngày"},
        {"Column": "Primary_Distraction_App", "Type": "Categorical (Nominal)", "Meaning": "Ứng dụng gây xao nhãng chính (TikTok, Instagram,...)"},
        {"Column": "Digital_Wellbeing_Feature_Used", "Type": "Categorical (Binary)", "Meaning": "Có sử dụng tính năng giới hạn thời gian (Screen Time) hay không"},
        {"Column": "Multitasking_Frequency", "Type": "Categorical (Ordinal)", "Meaning": "Tần suất làm nhiều việc số cùng lúc (Low, Medium, High)"},
        {"Column": "Productivity_Tools_Used", "Type": "Categorical (Nominal)", "Meaning": "Công cụ quản lý học tập sử dụng (Notion, Trello,...)"},
        {"Column": "Sleep_Duration_Hours", "Type": "Numerical (Continuous)", "Meaning": "Thời lượng giấc ngủ mỗi đêm (giờ)"},
        {"Column": "Physical_Activity_Minutes", "Type": "Numerical (Continuous)", "Meaning": "Thời gian hoạt động thể chất mỗi ngày (phút)"},
        {"Column": "Stress_Level", "Type": "Categorical (Ordinal)", "Meaning": "Mức độ căng thẳng tự đánh giá (Low, Medium, High)"},
        {"Column": "Attendance_Rate_Percentage", "Type": "Numerical (Continuous)", "Meaning": "Tỷ lệ điểm danh tham gia lớp học (%)"},
        {"Column": "Current_GPA", "Type": "Numerical (Continuous)", "Meaning": "Điểm trung bình tích lũy hiện tại (thang 4.0)"},
        {"Column": "Productivity_Score", "Type": "Numerical (Continuous Target)", "Meaning": "Điểm hiệu suất học tập tổng hợp (10 - 100)"}
    ]
    df_dict = pd.DataFrame(data_dict)
    df_dict.to_csv(os.path.join(metrics_dir, '01_data_dictionary.csv'), index=False, encoding='utf-8-sig')
    
    # 2. Raw Summary Statistics
    num_summary = df_raw[num_cols_raw].describe().T
    num_summary['skewness'] = df_raw[num_cols_raw].skew()
    num_summary['kurtosis'] = df_raw[num_cols_raw].kurtosis()
    num_summary.to_csv(os.path.join(metrics_dir, '01_numerical_summary_raw.csv'), encoding='utf-8-sig')
    
    cat_cols_raw = ['Gender', 'Academic_Level', 'Major_Category', 'Primary_Distraction_App',
                    'Digital_Wellbeing_Feature_Used', 'Multitasking_Frequency',
                    'Productivity_Tools_Used', 'Stress_Level']
    cat_summary = df_raw[cat_cols_raw].describe().T
    cat_summary.to_csv(os.path.join(metrics_dir, '01_categorical_summary_raw.csv'), encoding='utf-8-sig')
    
    # 3. Missing Values Diagnosis & Imputation
    missing_series = df_raw.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_series.index,
        'Missing_Count': missing_series.values,
        'Missing_Percentage': (missing_series.values / len(df_raw) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    missing_df.to_csv(os.path.join(metrics_dir, '01_missing_values_report.csv'), index=False, encoding='utf-8-sig')
    print(f"Missing values found in {len(missing_df)} columns.")
    
    df_clean = df_raw.copy()
    
    # Impute numerical missing values with median
    for c in num_cols_raw:
        if df_clean[c].isnull().sum() > 0:
            med_val = df_clean[c].median()
            df_clean[c] = df_clean[c].fillna(med_val)
            print(f"  Imputed {c} with median: {med_val:.2f}")
            
    # Impute categorical missing values with mode
    for c in cat_cols_raw:
        if df_clean[c].isnull().sum() > 0:
            mode_val = df_clean[c].mode()[0]
            df_clean[c] = df_clean[c].fillna(mode_val)
            print(f"  Imputed {c} with mode: {mode_val}")
            
    # 4. Outlier Analysis & Treatment (IQR & Z-score)
    outlier_records = []
    num_features = [c for c in num_cols_raw if c != 'Productivity_Score']
    
    for c in num_features:
        q1 = df_clean[c].quantile(0.25)
        q3 = df_clean[c].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        z_scores = np.abs(stats.zscore(df_clean[c]))
        outliers_iqr = ((df_clean[c] < lower_bound) | (df_clean[c] > upper_bound)).sum()
        outliers_z = (z_scores > 3.0).sum()
        
        outlier_records.append({
            'Feature': c,
            'Q1': round(q1, 2),
            'Q3': round(q3, 2),
            'IQR': round(iqr, 2),
            'Lower_Bound_IQR': round(lower_bound, 2),
            'Upper_Bound_IQR': round(upper_bound, 2),
            'Outliers_IQR_Count': outliers_iqr,
            'Outliers_Zscore_Count': outliers_z
        })
        
        # Capping / Winsorizing extreme outliers at IQR bounds
        df_clean[c] = np.clip(df_clean[c], lower_bound, upper_bound)
        
    df_outliers = pd.DataFrame(outlier_records)
    df_outliers.to_csv(os.path.join(metrics_dir, '01_outlier_summary.csv'), index=False, encoding='utf-8-sig')
    
    # 5. Target Discretization (Low, Medium, High)
    # Using 3-quantile binning (tertiles) for balanced class representation
    q_low, q_high = df_clean['Productivity_Score'].quantile([0.3333, 0.6667])
    print(f"Target Discretization Thresholds: Low <= {q_low:.2f} < Medium <= {q_high:.2f} < High")
    
    labels = ['Low', 'Medium', 'High']
    df_clean['Productivity_Class'] = pd.qcut(
        df_clean['Productivity_Score'],
        q=[0.0, 0.3333, 0.6667, 1.0],
        labels=labels
    )
    
    target_dist = df_clean['Productivity_Class'].value_counts().reset_index()
    target_dist.columns = ['Productivity_Class', 'Count']
    target_dist['Percentage'] = (target_dist['Count'] / len(df_clean) * 100).round(2)
    target_dist.to_csv(os.path.join(metrics_dir, '01_target_distribution.csv'), index=False, encoding='utf-8-sig')
    
    # 6. Feature Encoding
    # Ordinal features
    ordinal_mappings = {
        'Academic_Level': ['Freshman', 'Sophomore', 'Junior', 'Senior', 'Postgraduate'],
        'Multitasking_Frequency': ['Low', 'Medium', 'High'],
        'Stress_Level': ['Low', 'Medium', 'High']
    }
    
    df_encoded = df_clean.drop(columns=['Student_ID']).copy()
    
    for col, categories in ordinal_mappings.items():
        oe = OrdinalEncoder(categories=[categories])
        df_encoded[col + '_Ordinal'] = oe.fit_transform(df_encoded[[col]])
        df_encoded.drop(columns=[col], inplace=True)
        
    # Nominal features (One-Hot Encoding)
    nominal_cols = ['Gender', 'Major_Category', 'Primary_Distraction_App',
                    'Digital_Wellbeing_Feature_Used', 'Productivity_Tools_Used']
    
    ohe = OneHotEncoder(sparse_output=False, drop=None)
    ohe_matrix = ohe.fit_transform(df_encoded[nominal_cols])
    ohe_feature_names = ohe.get_feature_names_out(nominal_cols)
    df_ohe = pd.DataFrame(ohe_matrix, columns=ohe_feature_names, index=df_encoded.index)
    
    df_encoded = pd.concat([df_encoded.drop(columns=nominal_cols), df_ohe], axis=1)
    
    # Dimension comparison
    dim_before_ohe = len(df_clean.columns) - 2 # without Student_ID and Target
    dim_after_ohe = len(df_encoded.columns) - 2 # without Productivity_Score and Productivity_Class
    print(f"Feature Dimension Before OHE: {dim_before_ohe} -> After OHE: {dim_after_ohe}")
    
    # 7. Feature Normalization (StandardScaler)
    features_to_scale = [c for c in df_encoded.columns if c not in ['Productivity_Score', 'Productivity_Class']]
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_encoded[features_to_scale])
    df_scaled = pd.DataFrame(scaled_matrix, columns=features_to_scale, index=df_encoded.index)
    
    # Add back target
    df_processed = pd.concat([df_scaled, df_encoded[['Productivity_Score', 'Productivity_Class']]], axis=1)
    
    # 8. Export Processed Datasets & Summary
    processed_path = os.path.join(processed_dir, 'student_data_processed.csv')
    df_processed.to_csv(processed_path, index=False, encoding='utf-8-sig')
    
    processed_summary = df_processed[features_to_scale].describe().T
    processed_summary.to_csv(os.path.join(metrics_dir, '01_processed_summary.csv'), encoding='utf-8-sig')
    
    # Save dimension info
    dim_info = {
        "Total_Samples": len(df_processed),
        "Initial_Raw_Features": len(df_raw.columns),
        "Input_Features_Before_OHE": dim_before_ohe,
        "Input_Features_After_OHE": dim_after_ohe,
        "Scaled_Features_Count": len(features_to_scale),
        "Target_Classes": labels,
        "Low_Threshold": round(float(q_low), 2),
        "High_Threshold": round(float(q_high), 2)
    }
    with open(os.path.join(metrics_dir, '01_preprocessing_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(dim_info, f, indent=4, ensure_ascii=False)
        
    print(f"[SUCCESS] Step 1 Preprocessing completed. Processed file saved to '{processed_path}'")
    return df_processed

if __name__ == '__main__':
    run_preprocessing()
