import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_model(path):
    return joblib.load(path)


def results(df):
    st.subheader("Результаты предсказания")
    
    counts = df['predicted_rain'].value_counts().sort_index()
    labels = {0: 'Нет осадков (0)', 1: 'Есть осадки (1)'}
    x_labels = [labels.get(int(k), str(k)) for k in counts.index]
    values = counts.values
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x_labels, values, color=['#4C72B0', '#DD8452'])
    ax.set_ylabel('Количество дней')
    ax.set_title('Распределение предсказанных классов')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom')
    st.pyplot(fig)


def show_metrics(df):
    st.subheader("Метрики качества (на основе истинного rain)")
    y_true = df['rain'].astype(int)
    y_pred = df['predicted_rain']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1-score", f"{f1:.2%}")

    st.write("**Матрица ошибок**")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    im = ax_cm.imshow(cm, cmap='Blues')
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Pred 0', 'Pred 1'])
    ax_cm.set_yticklabels(['True 0', 'True 1'])
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax_cm.set_title('Confusion Matrix')
    fig_cm.colorbar(im, ax=ax_cm)
    st.pyplot(fig_cm)


def compare_pred_true(df):
    st.subheader("Сравнение кол-ва предсказанных и истинных значений в динамике")

    df_c = df.copy()
    df_c['date'] = pd.to_datetime(df['date']).dt.to_period('M')
        
    df_pred = df_c.groupby(df_c['date'])['predicted_rain'].sum().reset_index()
    df_true = df_c.groupby(df_c['date'])['rain'].sum().reset_index()

    max_months = len(df_pred)
    months_to_show = st.slider(
        "Сколько последних месяцев показать?", 
        min_value=6, max_value=max_months, value=min(36, max_months), step=6
    )

    df_pred = df_pred.tail(months_to_show)
    df_true = df_true.tail(months_to_show)

    fig_time, ax_time = plt.subplots(figsize=(12, 4))
    ax_time.plot(df_pred['date'].dt.to_timestamp(), df_pred['predicted_rain'], label='Predicted',
                color="#D31515", marker='o', markersize=3, linewidth=1)
    
    ax2 = ax_time.twinx()
    ax2.plot(df_true['date'].dt.to_timestamp(), df_true['rain'], label='True',
            color="#04398F", marker='o', markersize=3, linewidth=1)
    
    ax_time.set_ylabel('Количество осадков')
    ax_time.set_xlabel('Дата')
    ax_time.set_title('Динамика по месяцам')

    lines1, labels1 = ax_time.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_time.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig_time.autofmt_xdate()
    st.pyplot(fig_time)


def show_graph(df):
    st.subheader("Динамика по месяцам")


    df_c = df.copy()
    df_c['date'] = pd.to_datetime(df_c['date']).dt.to_period('M')

    df_grouped = df_c.groupby(df_c['date'])['predicted_rain'].sum().reset_index()

    max_months = len(df_grouped)
    months_to_show = st.slider(
        "Сколько последних месяцев показать?", 
        min_value=6, max_value=max_months, value=min(36, max_months), step=6
    )

    df_plot = df_grouped.tail(months_to_show)
    

    fig_time, ax_time = plt.subplots(figsize=(12, 4))
    ax_time.plot(df_plot['date'].dt.to_timestamp(), df_plot['predicted_rain'], label='Predicted',
                    color="#D31515", marker='o', markersize=4, linewidth=2)

    ax_time.set_ylabel('Количество осадков')
    ax_time.set_xlabel('Дата')
    ax_time.set_title(f'Динамика за последние {months_to_show} мес.')

    fig_time.autofmt_xdate()
    st.pyplot(fig_time)
    plt.close(fig_time)


st.title("Moscow Rain Prediction")

if 'df_result' not in st.session_state:
    st.session_state.df_result = None


model = load_model("model/laml_model.joblib")
st.success("Модель успешно загружена!")


if st.button("Сделать прогноз (predict)", type="primary"):

    file_path = 'data/topredict.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Файл не найден по пути: {file_path}")
        st.stop()

    required_cols = [f'precip_shift{i}' for i in range(1, 31)] + ['date']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"В файле не хватает столбцов: {missing}")
        st.stop()

    with st.spinner("Модель считает прогноз..."):
        pred_proba = model.predict(df).data[:, 0]

        df['predicted_proba'] = pred_proba
        df['predicted_rain'] = (pred_proba > 0.5).astype(int)

        st.session_state.df_result = df

if st.session_state.df_result is not None:
    df = st.session_state.df_result
    
    results(df)
    
    if 'rain' in df.columns:
        show_metrics(df)
        compare_pred_true(df)
    else:
        show_graph(df)
        

