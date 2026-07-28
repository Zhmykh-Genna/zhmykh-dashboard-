import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET
import time

# === Блок Google Trends с защитой ===
try:
    from pytrends.request import TrendReq
    PYTREENDS_AVAILABLE = True
except ImportError:
    PYTREENDS_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Макро-Дашборд + EUR/RUB")
st.title("📊 Макроэкономический дашборд: RUB, EUR, Нефть, DXY, Ставка ЦБ")

# === Боковая панель ===
st.sidebar.header("Настройки")
days = st.sidebar.slider("Период истории (дней)", 30, 730, 365, step=30)

st.sidebar.subheader("Параметры стратегии (MA, RSI)")
fast = st.sidebar.slider("Быстрая MA", 5, 50, 10, step=1)
slow = st.sidebar.slider("Медленная MA", 20, 200, 30, step=1)
rsi_period = st.sidebar.slider("Период RSI", 7, 21, 14, step=1)
rsi_overbought = st.sidebar.slider("RSI перекупленности (фильтр для BUY)", 60, 80, 70, step=1)

# === Функция загрузки ключевой ставки ЦБ (через официальный XML) ===
@st.cache_data(ttl=86400)
def load_cbr_key_rate():
    """Загружает историю ключевой ставки ЦБ РФ через XML-дамп с сайта ЦБ"""
    try:
        # Используем официальную XML-выгрузку ЦБ
        url = "https://www.cbr.ru/Queries/KeyRate?date=2020-01-01"
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        if response.status_code != 200:
            return None
        # Парсим XML
        root = ET.fromstring(response.text)
        # Ищем все элементы KeyRate
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
    except Exception as e:
        st.sidebar.warning(f"Ошибка загрузки ставки ЦБ: {e}")
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
    except Exception as e:
        st.sidebar.warning(f"Ошибка загрузки Google Trends: {e}")
        return None

# === Основная функция загрузки рыночных данных ===
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
    """Возвращает DataFrame с MA, RSI и сигналами (Filtered_Signal)"""
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

# === Загрузка данных ===
df = load_market_data(days)
if df is None:
    st.error("Не удалось загрузить рыночные данные. Проверьте интернет.")
    st.stop()

# Загружаем ключевую ставку и Google Trends
df_rate = load_cbr_key_rate()
df_trends = load_google_trends("курс доллара", min(days, 180))

# === Генерация сигналов ===
signals_usd = generate_signals(df['USD_RUB'], fast, slow, rsi_period, rsi_overbought)
signals_eur = generate_signals(df['EUR_RUB'], fast, slow, rsi_period, rsi_overbought)
signals_brent = generate_signals(df['Brent'], fast, slow, rsi_period, rsi_overbought)

df['DXY_RSI'] = calculate_rsi(df['DXY'], rsi_period)

# === Текущие значения ===
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last

# Метрики
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

# Ряд 5: Ключевая ставка ЦБ
if df_rate is not None and not df_rate.empty:
    fig.add_trace(go.Scatter(x=df_rate.index, y=df_rate['key_rate'], mode='lines+markers', name='Ключевая ставка', line=dict(color='red', width=2)), row=5, col=1)
    fig.update_yaxes(title_text="Ставка %", row=5, col=1)
else:
    # Если данных нет, выведем пустое место с сообщением
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
    st.info("Данные Google Trends временно недоступны. Это может быть связано с ограничениями сервиса или сетевых запросов из облачной среды. Остальные данные актуальны.")

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

st.caption("Данные: Yahoo Finance, ЦБ РФ (XML), Google Trends. Сигналы не являются инвестиционной рекомендацией.")import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import cbrapi
import time

# === Блок Google Trends с защитой ===
try:
    from pytrends.request import TrendReq
    PYTREENDS_AVAILABLE = True
except ImportError:
    PYTREENDS_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Макро-Дашборд + EUR/RUB")
st.title("📊 Макроэкономический дашборд: RUB, EUR, Нефть, DXY, Ставка ЦБ")

# === Боковая панель ===
st.sidebar.header("Настройки")
days = st.sidebar.slider("Период истории (дней)", 30, 730, 365, step=30)

st.sidebar.subheader("Параметры стратегии (MA, RSI)")
fast = st.sidebar.slider("Быстрая MA", 5, 50, 10, step=1)
slow = st.sidebar.slider("Медленная MA", 20, 200, 30, step=1)
rsi_period = st.sidebar.slider("Период RSI", 7, 21, 14, step=1)
rsi_overbought = st.sidebar.slider("RSI перекупленности (фильтр для BUY)", 60, 80, 70, step=1)

# === Функции загрузки ===
@st.cache_data(ttl=86400)
def load_cbr_key_rate():
    try:
        client = cbrapi.Client()
        data = client.get_key_rate()
        if data and hasattr(data, 'data'):
            df_rate = pd.DataFrame(data.data)
            df_rate['date'] = pd.to_datetime(df_rate['date'])
            df_rate.set_index('date', inplace=True)
            df_rate.columns = ['key_rate']
            return df_rate
    except:
        return None

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
    """Возвращает DataFrame с MA, RSI и сигналами (Filtered_Signal)"""
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

# === Загрузка данных ===
df = load_market_data(days)
if df is None:
    st.error("Не удалось загрузить данные.")
    st.stop()

df_rate = load_cbr_key_rate()
df_trends = load_google_trends("курс доллара", min(days, 180))

# === Генерация сигналов для USD, EUR, Brent ===
signals_usd = generate_signals(df['USD_RUB'], fast, slow, rsi_period, rsi_overbought)
signals_eur = generate_signals(df['EUR_RUB'], fast, slow, rsi_period, rsi_overbought)
signals_brent = generate_signals(df['Brent'], fast, slow, rsi_period, rsi_overbought)

# === RSI для DXY (только для отображения) ===
df['DXY_RSI'] = calculate_rsi(df['DXY'], rsi_period)

# === Текущие значения ===
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("USD/RUB", f"{last['USD_RUB']:.4f}", f"{(last['USD_RUB']/prev['USD_RUB']-1)*100:.2f}%")
col2.metric("EUR/RUB", f"{last['EUR_RUB']:.4f}", f"{(last['EUR_RUB']/prev['EUR_RUB']-1)*100:.2f}%")
col3.metric("Нефть Brent", f"{last['Brent']:.2f}", f"{(last['Brent']/prev['Brent']-1)*100:.2f}%")
col4.metric("Индекс DXY", f"{last['DXY']:.2f}", f"{(last['DXY']/prev['DXY']-1)*100:.2f}%")
col5.metric("Ключевая ставка ЦБ", f"{df_rate.iloc[-1]['key_rate']:.2f}%" if df_rate is not None else "—")
col6.metric("Сигнал USD", {1:"🟢 BUY", -1:"🔴 SELL", 0:"⚪ Нейтр"}.get(int(signals_usd.iloc[-1]['Filtered_Signal']) if not signals_usd.empty else 0, "—"))

# === Основной график с 5 рядами ===
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
# Отметки сигналов USD
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

# Ряд 3: Нефть Brent с сигналами
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

# Ряд 5: Ключевая ставка ЦБ
if df_rate is not None:
    fig.add_trace(go.Scatter(x=df_rate.index, y=df_rate['key_rate'], mode='lines+markers', name='Ключевая ставка', line=dict(color='red')), row=5, col=1)
fig.update_yaxes(title_text="Ставка %", row=5, col=1)

fig.update_layout(template='plotly_dark', height=1200, hovermode='x unified')
fig.update_xaxes(title_text="Дата", row=5, col=1)

st.plotly_chart(fig, use_container_width=True)

# === Google Trends ===
if df_trends is not None:
    st.subheader("📈 Google Trends: 'курс доллара'")
    st.line_chart(df_trends)

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

st.caption("Данные: Yahoo Finance, ЦБ РФ, Google Trends. Сигналы не являются инвестиционной рекомендацией.")