import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET
import numpy as np
from itertools import product

# === Блок Google Trends с защитой ===
try:
    from pytrends.request import TrendReq
    PYTREENDS_AVAILABLE = True
except ImportError:
    PYTREENDS_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Макро-Дашборд + Бэктест + Оптимизация")
st.title("📊 Макроэкономический дашборд + Автоматический бэктест и оптимизация")

# === Боковая панель ===
st.sidebar.header("Настройки")
days = st.sidebar.slider("Период истории (дней)", 30, 730, 365, step=30)

st.sidebar.subheader("Параметры стратегии (MA, RSI)")
fast = st.sidebar.slider("Быстрая MA", 5, 50, 10, step=1)
slow = st.sidebar.slider("Медленная MA", 20, 200, 30, step=1)
rsi_period = st.sidebar.slider("Период RSI", 7, 21, 14, step=1)
rsi_overbought = st.sidebar.slider("RSI перекупленности (фильтр для BUY)", 60, 80, 70, step=1)

st.sidebar.subheader("Настройки бэктеста")
backtest_asset = st.sidebar.selectbox("Актив для бэктеста", ["USD/RUB", "EUR/RUB", "Brent"])
initial_capital = st.sidebar.number_input("Начальный капитал (руб.)", min_value=1000, value=10000, step=1000)
commission = st.sidebar.slider("Комиссия на сделку (%)", 0.0, 1.0, 0.1, step=0.05) / 100

# === Секция оптимизации ===
st.sidebar.subheader("Оптимизация параметров")
optimize_enabled = st.sidebar.checkbox("Включить оптимизацию", value=False)
if optimize_enabled:
    st.sidebar.write("Диапазоны для перебора:")
    fast_range = st.sidebar.slider("Быстрая MA (мин, макс, шаг)", 5, 50, (5, 20, 5), step=1)
    slow_range = st.sidebar.slider("Медленная MA (мин, макс, шаг)", 20, 200, (20, 60, 10), step=1)
    rsi_period_range = st.sidebar.slider("Период RSI (мин, макс, шаг)", 7, 21, (7, 14, 7), step=1)
    rsi_ob_range = st.sidebar.slider("RSI перекупленности (мин, макс, шаг)", 60, 80, (65, 75, 5), step=1)
    optimize_metric = st.sidebar.selectbox("Критерий оптимизации", ["Доходность %", "Коэффициент Шарпа"])
    optimize_button = st.sidebar.button("Запустить оптимизацию")

# === Функция загрузки ключевой ставки ЦБ ===
@st.cache_data(ttl=86400)
def load_cbr_key_rate():
    try:
        url = "https://www.cbr.ru/Queries/KeyRate?date=2020-01-01"
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        if response.status_code != 200:
            return None
        root = ET.fromstring(response.text)
        rates = []
        for elem in root.findall(".//KeyRate"):
            date_str = elem.find('Date').text
            rate_str = elem.find('Rate').text
            if date_str and rate_str:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                rate = float(rate_str.replace(',', '.'))
                rates.append((date, rate))
        if not rates:
            return None
        df_rate = pd.DataFrame(rates, columns=['date', 'key_rate'])
        df_rate.set_index('date', inplace=True)
        df_rate.sort_index(inplace=True)
        return df_rate
    except:
        return None

# === Функция загрузки Google Trends ===
@st.cache_data(ttl=3600)
def load_google_trends(keyword, days):
    if not PYTREENDS_AVAILABLE:
        return None
    try:
        pytrends = TrendReq(hl='ru', tz=180)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        pytrends.build_payload(kw_list=[keyword], timeframe=timeframe, geo='RU')
        df = pytrends.interest_over_time()
        if df.empty:
            return None
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        df.columns = ['trend_index']
        return df
    except:
        return None

# === Загрузка рыночных данных ===
@st.cache_data(ttl=3600)
def load_market_data(days):
    end = datetime.now()
    start = end - timedelta(days=days)
    rub = yf.download("USDRUB=X", start=start, end=end, progress=False)
    eur = yf.download("EURRUB=X", start=start, end=end, progress=False)
    brent = yf.download("BZ=F", start=start, end=end, progress=False)
    dxy = yf.download("DX-Y.NYB", start=start, end=end, progress=False)
    if rub.empty or eur.empty or brent.empty or dxy.empty:
        return None
    df = pd.DataFrame(index=rub.index)
    df['USD_RUB'] = rub['Close']
    df['EUR_RUB'] = eur['Close']
    df['Brent'] = brent['Close']
    df['DXY'] = dxy['Close']
    df.dropna(inplace=True)
    return df

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(price_series, fast, slow, rsi_period, rsi_overbought):
    df = pd.DataFrame(index=price_series.index)
    df['price'] = price_series
    df['MAf'] = df['price'].rolling(fast).mean()
    df['MAs'] = df['price'].rolling(slow).mean()
    df['RSI'] = calculate_rsi(df['price'], rsi_period)
    df['Signal'] = 0
    df.loc[(df['MAf'] > df['MAs']) & (df['MAf'].shift(1) <= df['MAs'].shift(1)), 'Signal'] = 1
    df.loc[(df['MAf'] < df['MAs']) & (df['MAf'].shift(1) >= df['MAs'].shift(1)), 'Signal'] = -1
    df['Filtered_Signal'] = df['Signal'].copy()
    df.loc[(df['Signal'] == 1) & (df['RSI'] >= rsi_overbought), 'Filtered_Signal'] = 0
    df.dropna(inplace=True)
    return df

# === Функция бэктеста с учётом депозита и Шарпа ===
def run_backtest(price_series, signals_df, initial_capital=10000, commission=0.001, risk_free_rate=None):
    """
    risk_free_rate - средняя ключевая ставка за период (в долях, годовая), если None, не учитываем
    """
    if signals_df.empty or len(signals_df) < 2:
        return None
    
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = []

    for i in range(len(signals_df)):
        price = signals_df.iloc[i]['price']
        signal = signals_df.iloc[i]['Filtered_Signal']
        
        if signal == 1 and position == 0:
            position = capital / price
            entry_price = price
            capital = 0
            trades.append({'Дата входа': signals_df.index[i], 'Цена входа': price})
        
        elif signal == -1 and position > 0:
            exit_price = price
            gross = position * exit_price
            fee = gross * commission
            capital = gross - fee
            trade_return = (exit_price / entry_price - 1) * 100
            trades[-1]['Дата выхода'] = signals_df.index[i]
            trades[-1]['Цена выхода'] = exit_price
            trades[-1]['Доходность %'] = trade_return
            position = 0
        
        if position > 0:
            current_value = capital + position * price
        else:
            current_value = capital
        equity_curve.append(current_value)
    
    if position > 0 and trades:
        last_price = signals_df.iloc[-1]['price']
        gross = position * last_price
        fee = gross * commission
        capital = gross - fee
        trade_return = (last_price / trades[-1]['Цена входа'] - 1) * 100
        trades[-1]['Дата выхода'] = signals_df.index[-1]
        trades[-1]['Цена выхода'] = last_price
        trades[-1]['Доходность %'] = trade_return
        position = 0
        equity_curve[-1] = capital
    
    final_capital = capital if position == 0 else capital + position * signals_df.iloc[-1]['price']
    total_return = (final_capital / initial_capital - 1) * 100
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    if not trades_df.empty:
        win_trades = (trades_df['Доходность %'] > 0).sum()
        loss_trades = (trades_df['Доходность %'] < 0).sum()
        win_rate = win_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_return = trades_df['Доходность %'].mean()
        max_return = trades_df['Доходность %'].max()
        min_return = trades_df['Доходность %'].min()
    else:
        win_rate = avg_return = max_return = min_return = 0
    
    equity_series = pd.Series(equity_curve, index=signals_df.index)
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    
    # Доходность депозита (если ставка известна)
    deposit_return = None
    deposit_equity = None
    if risk_free_rate is not None:
        # Считаем, что ставка годовая, капитал растёт линейно (простые проценты)
        days_in_period = (signals_df.index[-1] - signals_df.index[0]).days
        deposit_return = risk_free_rate * (days_in_period / 365) * 100  # в процентах
        # Кривая капитала депозита (линейный рост от initial до final)
        deposit_equity = initial_capital * (1 + risk_free_rate * (np.arange(len(equity_series)) / 365))
        deposit_equity = pd.Series(deposit_equity, index=signals_df.index)
    
    # Коэффициент Шарпа (годовой, с учётом безрисковой ставки)
    returns = equity_series.pct_change().dropna()
    if len(returns) > 1 and risk_free_rate is not None:
        # Средняя дневная доходность стратегии
        mean_daily_return = returns.mean()
        std_daily_return = returns.std()
        # Безрисковая дневная ставка (годовая / 252)
        rf_daily = risk_free_rate / 252
        if std_daily_return != 0:
            sharpe = (mean_daily_return - rf_daily) / std_daily_return * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    return {
        'total_return': total_return,
        'final_capital': final_capital,
        'num_trades': len(trades_df),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'max_return': max_return,
        'min_return': min_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'trades_df': trades_df,
        'equity_curve': equity_series,
        'deposit_return': deposit_return,
        'deposit_equity': deposit_equity
    }

# === Функция оптимизации параметров ===
def optimize_parameters(price_series, fast_range, slow_range, rsi_period_range, rsi_ob_range, initial_capital, commission, risk_free_rate, metric='return'):
    best_result = None
    best_params = None
    best_value = -np.inf if metric == 'return' else -np.inf
    total_combinations = 0
    for fast, slow, rsi_p, rsi_ob in product(
        range(fast_range[0], fast_range[1]+1, fast_range[2]),
        range(slow_range[0], slow_range[1]+1, slow_range[2]),
        range(rsi_period_range[0], rsi_period_range[1]+1, rsi_period_range[2]),
        range(rsi_ob_range[0], rsi_ob_range[1]+1, rsi_ob_range[2])
    ):
        if fast >= slow:
            continue
        total_combinations += 1
        signals = generate_signals(price_series, fast, slow, rsi_p, rsi_ob)
        if signals.empty or len(signals) < 2:
            continue
        result = run_backtest(price_series, signals, initial_capital, commission, risk_free_rate)
        if result is None:
            continue
        # Критерий
        if metric == 'Доходность %':
            value = result['total_return']
        else:  # Шарп
            value = result['sharpe_ratio']
        if value > best_value:
            best_value = value
            best_result = result
            best_params = (fast, slow, rsi_p, rsi_ob)
    return best_params, best_result, total_combinations

# === Загрузка данных ===
df = load_market_data(days)
if df is None:
    st.error("Не удалось загрузить рыночные данные. Проверьте интернет.")
    st.stop()

df_rate = load_cbr_key_rate()
df_trends = load_google_trends("курс доллара", min(days, 180))

# Текущая ключевая ставка для безрисковой ставки
current_rate = df_rate.iloc[-1]['key_rate'] / 100 if df_rate is not None else 0.21  # запасное значение

# === Генерация сигналов ===
signals_usd = generate_signals(df['USD_RUB'], fast, slow, rsi_period, rsi_overbought)
signals_eur = generate_signals(df['EUR_RUB'], fast, slow, rsi_period, rsi_overbought)
signals_brent = generate_signals(df['Brent'], fast, slow, rsi_period, rsi_overbought)

df['DXY_RSI'] = calculate_rsi(df['DXY'], rsi_period)

# === Текущие значения ===
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("USD/RUB", f"{last['USD_RUB']:.4f}", f"{(last['USD_RUB']/prev['USD_RUB']-1)*100:.2f}%")
col2.metric("EUR/RUB", f"{last['EUR_RUB']:.4f}", f"{(last['EUR_RUB']/prev['EUR_RUB']-1)*100:.2f}%")
col3.metric("Нефть Brent", f"{last['Brent']:.2f}", f"{(last['Brent']/prev['Brent']-1)*100:.2f}%")
col4.metric("Индекс DXY", f"{last['DXY']:.2f}", f"{(last['DXY']/prev['DXY']-1)*100:.2f}%")
if df_rate is not None and not df_rate.empty:
    col5.metric("Ключевая ставка ЦБ", f"{df_rate.iloc[-1]['key_rate']:.2f}%")
else:
    col5.metric("Ключевая ставка ЦБ", "—", help="Данные не загружены")

if not signals_usd.empty:
    current_signal = int(signals_usd.iloc[-1]['Filtered_Signal'])
    signal_text = {1:"🟢 BUY", -1:"🔴 SELL", 0:"⚪ Нейтр"}[current_signal]
else:
    signal_text = "—"
col6.metric("Сигнал USD", signal_text)

# === Основной график ===
fig = make_subplots(
    rows=5, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.25, 0.25, 0.2, 0.15, 0.15]
)

# Ряд 1: USD/RUB
fig.add_trace(go.Scatter(x=df.index, y=df['USD_RUB'], mode='lines', name='USD/RUB', line=dict(color='white')), row=1, col=1)
fig.add_trace(go.Scatter(x=signals_usd.index, y=signals_usd['MAf'], mode='lines', name=f'MA{fast}', line=dict(color='orange', dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=signals_usd.index, y=signals_usd['MAs'], mode='lines', name=f'MA{slow}', line=dict(color='cyan', dash='dot')), row=1, col=1)
buy_usd = signals_usd[signals_usd['Filtered_Signal'] == 1]
sell_usd = signals_usd[signals_usd['Filtered_Signal'] == -1]
if not buy_usd.empty:
    fig.add_trace(go.Scatter(x=buy_usd.index, y=buy_usd['price'], mode='markers', name='BUY USD', marker=dict(color='green', size=8, symbol='triangle-up')), row=1, col=1)
if not sell_usd.empty:
    fig.add_trace(go.Scatter(x=sell_usd.index, y=sell_usd['price'], mode='markers', name='SELL USD', marker=dict(color='red', size=8, symbol='triangle-down')), row=1, col=1)
fig.update_yaxes(title_text="USD/RUB", row=1, col=1)

# Ряд 2: EUR/RUB
fig.add_trace(go.Scatter(x=df.index, y=df['EUR_RUB'], mode='lines', name='EUR/RUB', line=dict(color='lightblue')), row=2, col=1)
fig.add_trace(go.Scatter(x=signals_eur.index, y=signals_eur['MAf'], mode='lines', name=f'MA{fast} EUR', line=dict(color='orange', dash='dot')), row=2, col=1)
fig.add_trace(go.Scatter(x=signals_eur.index, y=signals_eur['MAs'], mode='lines', name=f'MA{slow} EUR', line=dict(color='cyan', dash='dot')), row=2, col=1)
buy_eur = signals_eur[signals_eur['Filtered_Signal'] == 1]
sell_eur = signals_eur[signals_eur['Filtered_Signal'] == -1]
if not buy_eur.empty:
    fig.add_trace(go.Scatter(x=buy_eur.index, y=buy_eur['price'], mode='markers', name='BUY EUR', marker=dict(color='green', size=8, symbol='triangle-up')), row=2, col=1)
if not sell_eur.empty:
    fig.add_trace(go.Scatter(x=sell_eur.index, y=sell_eur['price'], mode='markers', name='SELL EUR', marker=dict(color='red', size=8, symbol='triangle-down')), row=2, col=1)
fig.update_yaxes(title_text="EUR/RUB", row=2, col=1)

# Ряд 3: Нефть Brent
fig.add_trace(go.Scatter(x=df.index, y=df['Brent'], mode='lines', name='Brent', line=dict(color='orange')), row=3, col=1)
fig.add_trace(go.Scatter(x=signals_brent.index, y=signals_brent['MAf'], mode='lines', name=f'MA{fast} Brent', line=dict(color='red', dash='dot')), row=3, col=1)
fig.add_trace(go.Scatter(x=signals_brent.index, y=signals_brent['MAs'], mode='lines', name=f'MA{slow} Brent', line=dict(color='purple', dash='dot')), row=3, col=1)
buy_br = signals_brent[signals_brent['Filtered_Signal'] == 1]
sell_br = signals_brent[signals_brent['Filtered_Signal'] == -1]
if not buy_br.empty:
    fig.add_trace(go.Scatter(x=buy_br.index, y=buy_br['price'], mode='markers', name='BUY Brent', marker=dict(color='green', size=8, symbol='triangle-up')), row=3, col=1)
if not sell_br.empty:
    fig.add_trace(go.Scatter(x=sell_br.index, y=sell_br['price'], mode='markers', name='SELL Brent', marker=dict(color='red', size=8, symbol='triangle-down')), row=3, col=1)
fig.update_yaxes(title_text="Нефть", row=3, col=1)

# Ряд 4: DXY
fig.add_trace(go.Scatter(x=df.index, y=df['DXY'], mode='lines', name='DXY', line=dict(color='green')), row=4, col=1)
fig.update_yaxes(title_text="DXY", row=4, col=1)

# Ряд 5: Ключевая ставка
if df_rate is not None and not df_rate.empty:
    fig.add_trace(go.Scatter(x=df_rate.index, y=df_rate['key_rate'], mode='lines+markers', name='Ключевая ставка', line=dict(color='red', width=2)), row=5, col=1)
    fig.update_yaxes(title_text="Ставка %", row=5, col=1)
else:
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Нет данных'), row=5, col=1)
    fig.update_yaxes(title_text="Ставка % (недоступна)", row=5, col=1)

fig.update_layout(template='plotly_dark', height=1200, hovermode='x unified')
fig.update_xaxes(title_text="Дата", row=5, col=1)

st.plotly_chart(fig, use_container_width=True)

# === Google Trends ===
st.subheader("🔍 Google Trends: 'курс доллара' в России")
if df_trends is not None and not df_trends.empty:
    st.line_chart(df_trends)
else:
    st.info("Данные Google Trends временно недоступны.")

# === БЭКТЕСТ ===
st.subheader("📈 Автоматический бэктест стратегии")

# Выбор данных для бэктеста
if backtest_asset == "USD/RUB":
    price_series = df['USD_RUB']
    signals = signals_usd
elif backtest_asset == "EUR/RUB":
    price_series = df['EUR_RUB']
    signals = signals_eur
else:  # Brent
    price_series = df['Brent']
    signals = signals_brent

# Запуск бэктеста
if not signals.empty and len(signals) > 1:
    risk_free_rate = current_rate  # текущая ставка ЦБ как безрисковая
    backtest_result = run_backtest(price_series, signals, initial_capital, commission, risk_free_rate)
    if backtest_result:
        # Метрики в колонках
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Общая доходность", f"{backtest_result['total_return']:.2f}%")
        col2.metric("Конечный капитал", f"{backtest_result['final_capital']:.2f} ₽")
        col3.metric("Количество сделок", backtest_result['num_trades'])
        col4.metric("Процент прибыльных", f"{backtest_result['win_rate']:.1f}%")
        col5.metric("Макс. просадка", f"{backtest_result['max_drawdown']:.2f}%")
        col6.metric("Коэф. Шарпа (с учётом ставки)", f"{backtest_result['sharpe_ratio']:.2f}")

        st.write(f"**Средняя доходность на сделку:** {backtest_result['avg_return']:.2f}%  |  Лучшая: {backtest_result['max_return']:.2f}%  |  Худшая: {backtest_result['min_return']:.2f}%")

        # Сравнение с депозитом
        if backtest_result['deposit_return'] is not None:
            deposit_return = backtest_result['deposit_return']
            st.write(f"**Доходность депозита (под {risk_free_rate*100:.2f}% годовых):** {deposit_return:.2f}%")
            diff = backtest_result['total_return'] - deposit_return
            st.write(f"**Превышение стратегии над депозитом:** {diff:.2f}%")

        # График equity с депозитом
        st.subheader("📊 Кривая капитала vs депозит")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=backtest_result['equity_curve'].index, y=backtest_result['equity_curve'], mode='lines', name='Стратегия', line=dict(color='cyan')))
        if backtest_result['deposit_equity'] is not None:
            fig_equity.add_trace(go.Scatter(x=backtest_result['deposit_equity'].index, y=backtest_result['deposit_equity'], mode='lines', name='Депозит', line=dict(color='orange', dash='dot')))
        fig_equity.update_layout(template='plotly_dark', height=400, hovermode='x unified')
        st.plotly_chart(fig_equity, use_container_width=True)

        # Таблица сделок
        if not backtest_result['trades_df'].empty:
            with st.expander("📋 Детали сделок"):
                st.dataframe(backtest_result['trades_df'].style.format({
                    'Цена входа': '{:.4f}',
                    'Цена выхода': '{:.4f}',
                    'Доходность %': '{:.2f}'
                }))
    else:
        st.warning("Недостаточно данных для бэктеста (нужно минимум 2 сигнала).")
else:
    st.warning("Сигналов не обнаружено. Измените параметры стратегии, чтобы получить сигналы.")

# === ОПТИМИЗАЦИЯ ===
if optimize_enabled and 'optimize_button' in locals() and optimize_button:
    with st.spinner("Идёт перебор параметров... Это может занять до 1 минуты."):
        # Определяем диапазоны из слайдеров
        fast_min, fast_max, fast_step = fast_range
        slow_min, slow_max, slow_step = slow_range
        rsi_p_min, rsi_p_max, rsi_p_step = rsi_period_range
        rsi_ob_min, rsi_ob_max, rsi_ob_step = rsi_ob_range
        best_params, best_result, total_combos = optimize_parameters(
            price_series,
            (fast_min, fast_max, fast_step),
            (slow_min, slow_max, slow_step),
            (rsi_p_min, rsi_p_max, rsi_p_step),
            (rsi_ob_min, rsi_ob_max, rsi_ob_step),
            initial_capital,
            commission,
            current_rate,
            metric=optimize_metric
        )
        if best_params is not None and best_result is not None:
            st.success(f"Оптимизация завершена! Перебрано {total_combos} комбинаций.")
            st.write(f"**Лучшие параметры:** Быстрая MA = {best_params[0]}, Медленная MA = {best_params[1]}, Период RSI = {best_params[2]}, RSI перекупленности = {best_params[3]}")
            st.write(f"**Значение критерия ({optimize_metric}):** {best_result['total_return'] if optimize_metric == 'Доходность %' else best_result['sharpe_ratio']:.2f}")
            st.write(f"**Общая доходность:** {best_result['total_return']:.2f}%")
            st.write(f"**Коэф. Шарпа:** {best_result['sharpe_ratio']:.2f}")
            st.write(f"**Количество сделок:** {best_result['num_trades']}")
            st.write(f"**Макс. просадка:** {best_result['max_drawdown']:.2f}%")
            # Показать график equity для лучших параметров
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(x=best_result['equity_curve'].index, y=best_result['equity_curve'], mode='lines', name='Оптимальная стратегия', line=dict(color='lime')))
            if best_result['deposit_equity'] is not None:
                fig_opt.add_trace(go.Scatter(x=best_result['deposit_equity'].index, y=best_result['deposit_equity'], mode='lines', name='Депозит', line=dict(color='orange', dash='dot')))
            fig_opt.update_layout(template='plotly_dark', height=400, title='Кривая капитала для оптимальных параметров', hovermode='x unified')
            st.plotly_chart(fig_opt, use_container_width=True)
        else:
            st.warning("Не найдено подходящих комбинаций. Попробуйте расширить диапазоны.")

# === Корреляции ===
st.subheader("📈 Корреляция между активами")
periods = {'30 дней': 30, '90 дней': 90}
cols = st.columns(len(periods))
for i, (label, period) in enumerate(periods.items()):
    if len(df) > period:
        sub = df.iloc[-period:][['USD_RUB', 'EUR_RUB', 'Brent', 'DXY']].pct_change().dropna()
        corr = sub.corr()
        cols[i].write(f"**{label}**")
        cols[i].dataframe(corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1))

st.caption("Данные: Yahoo Finance, ЦБ РФ (XML), Google Trends. Сигналы, бэктест и оптимизация не являются инвестиционной рекомендацией.")
