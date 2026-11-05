import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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


def show_graph(df):
    st.subheader("Динамика по месяцам")


    df_c = df.copy()
    df_c['date'] = pd.to_datetime(df_c['date']).dt.to_period('M')

    df_grouped = df_c.groupby(df_c['date'])['predicted_rain'].sum().reset_index()

    max_months = len(df_grouped)
    if max_months < 6:
        months_to_show = max_months
    else:
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




def show_graph2(df):
    st.subheader("Динамика по месяцам с выбором периода")

    df_c = df.copy()
    df_c['date'] = pd.to_datetime(df_c['date']).dt.to_period('M')

    df_grouped = df_c.groupby(df_c['date'])['predicted_rain'].sum().reset_index()
    df_grouped = df_grouped.sort_values('date')

    min_month = df_grouped['date'].min()
    max_month = df_grouped['date'].max()

    months = pd.period_range(min_month, max_month, freq='M')
    month_labels = [str(month) for month in months]

    if len(month_labels) < 2:
        st.warning("Для построения графика необходимо минимум 2 месяца.")
        return

    default_range = (month_labels[0], month_labels[-1])

    if (
        'last_month_range_v2' not in st.session_state
        or any(
            month not in month_labels
            for month in st.session_state.last_month_range_v2
        )
    ):
        st.session_state.last_month_range_v2 = default_range

    def update_range():
        start, end = st.session_state.month_range_v2

        start_idx = month_labels.index(start)
        end_idx = month_labels.index(end)

        if end_idx - start_idx < 1:
            st.session_state.month_range_v2 = (
                st.session_state.last_month_range_v2
            )
        else:
            st.session_state.last_month_range_v2 = (start, end)

    start_month, end_month = st.select_slider(
        "Выберите период",
        options=month_labels,
        value=default_range,
        key="month_range_v2",
        on_change=update_range
    )

    start_period = pd.Period(start_month, freq='M')
    end_period = pd.Period(end_month, freq='M')

    st.write(f"**Период:** {start_month} — {end_month}")

    df_plot = df_grouped[
        (df_grouped['date'] >= start_period) &
        (df_grouped['date'] <= end_period)
    ]

    fig_time, ax_time = plt.subplots(figsize=(12, 4))
    ax_time.plot(df_plot['date'].dt.to_timestamp(), df_plot['predicted_rain'], label='Predicted',
                    color="#D31515", marker='o', markersize=4, linewidth=2)

    ax_time.set_ylabel('Количество осадков')
    ax_time.set_xlabel('Дата')
    ax_time.set_title(f'Динамика с {start_month} по {end_month}')

    fig_time.autofmt_xdate()
    st.pyplot(fig_time)
    plt.close(fig_time)





def show_streaks(df):
    st.subheader("Самые длинные периоды с осадками и без осадков")

    df_c = df.copy()
    df_c['date'] = pd.to_datetime(df_c['date'])
    df_c = df_c.sort_values('date').drop_duplicates('date')

    new_period = ( df_c['predicted_rain'].ne(df_c['predicted_rain'].shift()) | df_c['date'].diff().ne(pd.Timedelta(days=1)) )
    df_c['period'] = new_period.cumsum()

    df_periods = df_c.groupby('period').agg(
        predicted_rain=('predicted_rain', 'first'),
        start=('date', 'min'),
        end=('date', 'max'),
        days=('date', 'size')).reset_index(drop=True)

    col1, col2 = st.columns(2)
    for predicted_class, col, label in [(0, col1, 'Макс. дней без осадков'), (1, col2, 'Макс. дней с осадками')]:
        df_class = df_periods[df_periods['predicted_rain'] == predicted_class]

        if not df_class.empty:
            longest = df_class.loc[df_class['days'].idxmax()]
            col.metric(label, int(longest['days']))
            col.write(f"{longest['start']:%d.%m.%Y} — "f"{longest['end']:%d.%m.%Y}")
        else:
            col.metric(label, 0)

    st.write("**Топ-10 самых длинных периодов**")

    df_top = df_periods.sort_values('days', ascending=False).head(10).copy()
    df_top['predicted_rain'] = df_top['predicted_rain'].map({0: 'Без осадков (0)', 1: 'С осадками (1)'})
    df_top['start'] = df_top['start'].dt.strftime('%d.%m.%Y')
    df_top['end'] = df_top['end'].dt.strftime('%d.%m.%Y')
    df_top.columns = ['Тип периода', 'Начало', 'Конец', 'Дней']

    st.dataframe(df_top, hide_index=True)


def show_yearly_stats(df):
    st.subheader("Доля дней с осадками по годам")

    df_c = df.copy()
    df_c['date'] = pd.to_datetime(df_c['date'])
    df_c['year'] = df_c['date'].dt.year

    df_yearly = df_c.groupby('year')['predicted_rain'].agg(['sum', 'count']).reset_index()
    df_yearly['rain_percent'] = (df_yearly['sum'] / df_yearly['count'] * 100)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df_yearly['year'].astype(str), df_yearly['rain_percent'], color='#4C72B0')
    ax.set_ylabel('Доля дней с осадками, %')
    ax.set_xlabel('Год')
    ax.set_title('Доля предсказанных дней с осадками по годам')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)
    plt.close(fig)



st.title("Moscow Rain Prediction")

if 'df_result' not in st.session_state:
    st.session_state.df_result = None


if st.button("Показать результаты", type="primary"):

    file_path = 'data/predictions.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Файл не найден по пути: {file_path}")
        st.stop()

    required_cols = ['predicted_rain', 'date']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"В файле не хватает столбцов: {missing}")
        st.stop()

    if df.empty:
        st.error("Файл predictions.csv пуст.")
        st.stop()

    st.session_state.df_result = df

if st.session_state.df_result is not None:
    df = st.session_state.df_result
    
    results(df)
    show_graph(df)
    show_graph2(df)
    show_streaks(df)
    show_yearly_stats(df)
