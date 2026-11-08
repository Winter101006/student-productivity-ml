"""
Data Generator for Student Productivity and Digital Distraction Dataset
Generates a realistic, highly correlated dataset representing student academic life,
digital usage, sleep, stress, and productivity.
"""

import numpy as np
import pandas as pd
import os

def generate_dataset(n_samples=3500, random_state=42):
    np.random.seed(random_state)
    
    # 1. Demographics & Academics
    age = np.random.choice(np.arange(18, 28), size=n_samples, p=[0.15, 0.22, 0.24, 0.18, 0.10, 0.05, 0.03, 0.01, 0.01, 0.01])
    gender = np.random.choice(['Female', 'Male', 'Non-Binary'], size=n_samples, p=[0.50, 0.46, 0.04])
    
    academic_level = np.random.choice(
        ['Freshman', 'Sophomore', 'Junior', 'Senior', 'Postgraduate'],
        size=n_samples,
        p=[0.25, 0.28, 0.24, 0.18, 0.05]
    )
    
    major_category = np.random.choice(
        ['STEM', 'Business', 'Humanities', 'Health_Sciences', 'Arts_Design'],
        size=n_samples,
        p=[0.35, 0.25, 0.18, 0.12, 0.10]
    )
    
    # 2. Digital Habits & Behaviors (correlated)
    # Base digital engagement factor (latent variable ~ N(0, 1))
    digital_addiction_latent = np.random.normal(0, 1, size=n_samples)
    
    # Daily Screen Time (Hours) ~ depends on latent
    screen_time = 4.5 + 2.5 * (digital_addiction_latent - digital_addiction_latent.min()) / (digital_addiction_latent.max() - digital_addiction_latent.min()) * 4 + np.random.normal(0, 0.8, size=n_samples)
    screen_time = np.clip(screen_time, 2.5, 15.0)
    
    # Social Media Hours (strongly tied to screen time)
    social_media_ratio = np.random.beta(3, 4, size=n_samples)
    social_media_hours = np.clip(screen_time * social_media_ratio * 0.7 + np.random.normal(0, 0.3, size=n_samples), 0.2, 8.0)
    
    # Gaming Hours
    gaming_hours = np.random.exponential(scale=1.2, size=n_samples)
    gaming_hours = np.clip(gaming_hours + 0.4 * digital_addiction_latent, 0.0, 7.0)
    
    # Phone Unlocks Per Day (correlated with screen time and distraction)
    phone_unlocks = np.clip(40 + screen_time * 8.5 + np.random.normal(0, 15, size=n_samples), 20, 220).astype(int)
    
    # Daily Notifications
    notifications = np.clip(phone_unlocks * 1.3 + np.random.normal(20, 20, size=n_samples), 30, 320).astype(int)
    
    # Primary Distraction App
    app_choices = ['TikTok', 'Instagram', 'YouTube', 'Facebook', 'Discord', 'Video_Games']
    app_probs = [0.32, 0.28, 0.20, 0.08, 0.07, 0.05]
    primary_app = np.random.choice(app_choices, size=n_samples, p=app_probs)
    
    # Digital Wellbeing Feature Used
    wellbeing_prob = 1 / (1 + np.exp(0.3 * digital_addiction_latent)) # Less addicted use it more, or aware students
    digital_wellbeing = np.where(np.random.rand(n_samples) < 0.42, 'Yes', 'No')
    
    # Multitasking Frequency
    multitask_latent = 0.5 * digital_addiction_latent + np.random.normal(0, 0.6, size=n_samples)
    multitasking = np.where(multitask_latent < -0.3, 'Low', np.where(multitask_latent < 0.5, 'Medium', 'High'))
    
    # Productivity Tools Used
    tools = ['Notion', 'Forest_Pomodoro', 'Google_Workspace', 'Trello', 'None']
    tool_probs = [0.28, 0.22, 0.24, 0.08, 0.18]
    productivity_tool = np.random.choice(tools, size=n_samples, p=tool_probs)
    
    # 3. Health & Well-being
    # Sleep duration negatively affected by screen time and gaming
    sleep_hours = 8.5 - 0.22 * screen_time - 0.15 * gaming_hours + np.random.normal(0, 0.6, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 3.5, 10.5)
    
    # Physical activity (minutes per day)
    physical_activity = np.clip(60 - 3.2 * screen_time + np.random.exponential(25, size=n_samples), 0, 150).astype(int)
    
    # Stress level (Low, Medium, High)
    stress_latent = 0.35 * screen_time - 0.25 * sleep_hours + 0.3 * (multitasking == 'High') + np.random.normal(0, 0.7, size=n_samples)
    stress_level = np.where(stress_latent < 0.8, 'Low', np.where(stress_latent < 2.0, 'Medium', 'High'))
    
    # 4. Academic Performance & Target
    # Study hours per day
    study_hours = np.clip(6.5 - 0.25 * screen_time + 0.3 * (academic_level == 'Senior') + 0.5 * (academic_level == 'Postgraduate') + np.random.normal(0, 1.1, size=n_samples), 0.5, 10.0)
    
    # Attendance Rate (%)
    attendance_rate = np.clip(92 - 1.8 * screen_time + 1.2 * sleep_hours + np.random.normal(0, 4.5, size=n_samples), 45.0, 100.0)
    
    # Current GPA (scale 4.0)
    gpa = 2.2 + 0.18 * study_hours - 0.05 * social_media_hours + 0.012 * attendance_rate + 0.08 * (productivity_tool != 'None') + np.random.normal(0, 0.25, size=n_samples)
    gpa = np.clip(gpa, 1.5, 4.0)
    
    # Productivity Score (10.0 - 100.0)
    # Defined by positive factors: study hours, sleep, GPA, attendance, healthy habits
    # and negative factors: excessive screen time, social media distraction, gaming, high stress
    tool_bonus = np.array([4.0 if t in ['Notion', 'Forest_Pomodoro', 'Google_Workspace'] else 0.0 for t in productivity_tool])
    wellbeing_bonus = np.where(digital_wellbeing == 'Yes', 2.5, 0.0)
    stress_penalty = np.where(stress_level == 'High', 3.5, 0.0)
    
    productivity_score = (
        25.0
        + 4.2 * study_hours
        + 8.5 * gpa
        + 0.15 * attendance_rate
        + 1.8 * sleep_hours
        + 0.05 * physical_activity
        - 1.8 * social_media_hours
        - 1.2 * gaming_hours
        - 0.06 * phone_unlocks
        + tool_bonus
        + wellbeing_bonus
        - stress_penalty
        + np.random.normal(0, 3.5, size=n_samples)
    )
    productivity_score = np.clip(productivity_score, 12.0, 98.0)
    
    student_id = [f"STU_{10000 + i}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        'Student_ID': student_id,
        'Age': age,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Major_Category': major_category,
        'Daily_Screen_Time_Hours': np.round(screen_time, 2),
        'Study_Hours_Per_Day': np.round(study_hours, 2),
        'Social_Media_Hours': np.round(social_media_hours, 2),
        'Gaming_Hours': np.round(gaming_hours, 2),
        'Phone_Unlocks_Per_Day': phone_unlocks,
        'Notification_Count_Daily': notifications,
        'Primary_Distraction_App': primary_app,
        'Digital_Wellbeing_Feature_Used': digital_wellbeing,
        'Multitasking_Frequency': multitasking,
        'Productivity_Tools_Used': productivity_tool,
        'Sleep_Duration_Hours': np.round(sleep_hours, 2),
        'Physical_Activity_Minutes': physical_activity,
        'Stress_Level': stress_level,
        'Attendance_Rate_Percentage': np.round(attendance_rate, 2),
        'Current_GPA': np.round(gpa, 2),
        'Productivity_Score': np.round(productivity_score, 2)
    })
    
    # 5. Inject realistic missing values (~1-2% in some fields) to test Preprocessing pipeline
    nan_mask_study = np.random.rand(n_samples) < 0.015
    df.loc[nan_mask_study, 'Study_Hours_Per_Day'] = np.nan
    
    nan_mask_sleep = np.random.rand(n_samples) < 0.012
    df.loc[nan_mask_sleep, 'Sleep_Duration_Hours'] = np.nan
    
    nan_mask_att = np.random.rand(n_samples) < 0.010
    df.loc[nan_mask_att, 'Attendance_Rate_Percentage'] = np.nan
    
    nan_mask_stress = np.random.rand(n_samples) < 0.010
    df.loc[nan_mask_stress, 'Stress_Level'] = np.nan
    
    nan_mask_major = np.random.rand(n_samples) < 0.008
    df.loc[nan_mask_major, 'Major_Category'] = np.nan
    
    # 6. Inject a few realistic outliers in digital usage (e.g. extreme phone unlockers or screen time)
    outlier_idx = np.random.choice(n_samples, size=15, replace=False)
    df.loc[outlier_idx[:5], 'Daily_Screen_Time_Hours'] = np.random.uniform(20.5, 23.5, size=5)
    df.loc[outlier_idx[5:10], 'Phone_Unlocks_Per_Day'] = np.random.randint(380, 520, size=5)
    df.loc[outlier_idx[10:], 'Gaming_Hours'] = np.random.uniform(14.0, 18.0, size=5)
    
    return df

if __name__ == '__main__':
    raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, 'student_productivity_distraction.csv')
    df = generate_dataset(n_samples=3500, random_state=42)
    df.to_csv(raw_path, index=False)
    print(f"[SUCCESS] Generated dataset with {df.shape[0]} rows and {df.shape[1]} columns at '{raw_path}'")
