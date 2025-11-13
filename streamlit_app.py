"""
app.py — Centered HTML Refresh button + responsive grid (single-file)

Summary:
- Refresh button rendered in HTML for pixel control; clicking it navigates to ?refresh=<ts>.
- App detects ?refresh, clears cache, removes the param and reruns.
- Cards rendered as one responsive HTML grid via components.html; computed height prevents clipping.
- Left sidebar, CSV button, cache banners and footer removed.

Run:
    pip install streamlit yfinance pandas numpy
    streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import time
import streamlit.components.v1 as components

# -------------------------
# Config
# -------------------------
RSI_PERIOD = 14
VOL_DAYS = 252
CACHE_TTL_SECONDS = 600
LOOKBACK = "2y"
N_CARDS = 10
CARD_HEIGHT = 170  # px (used to compute iframe height)

# NIFTY50 tickers (common set)
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","KOTAKBANK.NS",
    "LT.NS","ITC.NS","SBIN.NS","HCLTECH.NS","AXISBANK.NS","BHARTIARTL.NS","BAJAJ-AUTO.NS",
    "ASIANPAINT.NS","HINDUNILVR.NS","MARUTI.NS","SUNPHARMA.NS","NTPC.NS","M&M.NS",
    "BAJFINANCE.NS","ULTRACEMCO.NS","TITAN.NS","POWERGRID.NS","ONGC.NS","HDFCLIFE.NS",
    "NESTLEIND.NS","DIVISLAB.NS","WIPRO.NS","BRITANNIA.NS","TECHM.NS","COALINDIA.NS",
    "SBILIFE.NS","ADANIENT.NS","ADANIPORTS.NS","TATASTEEL.NS","BPCL.NS","INDUSINDBK.NS",
    "GRASIM.NS","EICHERMOT.NS","JSWSTEEL.NS","DRREDDY.NS","TATAMOTORS.NS","CIPLA.NS",
    "SHREECEM.NS","HINDALCO.NS"
]

PALETTE = {
    "accent": "#2B8AEB",
    "green": "#2ecc71",
    "red": "#e74c3c",
    "yellow": "#fff7cc",
    "gray": "#95a5a6",
    "text": "#0F172A",
    "muted": "#6B7280",
    "shadow": "rgba(2,6,23,0.06)"
}

# -------------------------
# Financial helpers
# -------------------------
def rsi(series: pd.Series, period=RSI_PERIOD) -> pd.Series:
    s = series.dropna().astype(float)
    if s.empty:
        return s
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/period, adjust=False).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_up / avg_down
    r = 100 - (100 / (1 + rs))
    return r

def annual_vol(close_series: pd.Series) -> float:
    s = close_series.dropna().astype(float)
    if len(s) < 10:
        return float("nan")
    s = s.iloc[-VOL_DAYS:] if len(s) >= VOL_DAYS else s
    log_ret = np.log(s / s.shift(1)).dropna()
    if len(log_ret) < 2:
        return float("nan")
    return float(log_ret.std(ddof=0) * np.sqrt(252))

# -------------------------
# Data fetch (cached)
# -------------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_all(tickers):
    try:
        raw = yf.download(tickers, period=LOOKBACK, interval="1d", group_by="ticker", threads=True, progress=False)
    except Exception as e:
        st.session_state["_last_fetch_error"] = str(e)
        return {}
    out = {}
    if len(tickers) == 1:
        out[tickers[0]] = raw
        return out
    for t in tickers:
        try:
            df = raw[t].dropna(how='all').copy()
            df.index = pd.to_datetime(df.index)
            out[t] = df
        except Exception:
            try:
                df2 = yf.download(t, period=LOOKBACK, interval="1d", progress=False)
                df2.index = pd.to_datetime(df2.index)
                out[t] = df2
            except Exception:
                out[t] = pd.DataFrame()
    return out

# -------------------------
# Analysis: pick N_CARDS least-vol tickers
# -------------------------
def analyze_universe():
    price_map = fetch_all(NIFTY50)
    vol_list = []
    for tk, df in price_map.items():
        if df is None or df.empty or 'Close' not in df.columns:
            continue
        vol_list.append((tk, annual_vol(df['Close'])))
    vol_list = [x for x in vol_list if not np.isnan(x[1])]
    vol_list.sort(key=lambda x: x[1])
    selected = []
    for tk, vol in vol_list:
        if len(selected) >= N_CARDS:
            break
        df = price_map.get(tk, pd.DataFrame())
        if df is None or df.empty or 'Close' not in df.columns:
            continue
        daily = df['Close'].dropna()
        if len(daily) < 40:
            continue
        weekly = df.resample('W').last()['Close'].dropna()
        if len(weekly) < 8:
            continue
        try:
            d = float(rsi(daily).iloc[-1])
            w = float(rsi(weekly).iloc[-1])
        except Exception:
            continue
        # company name best-effort
        name = ""
        try:
            info = yf.Ticker(tk).info
            name = info.get("shortName") or info.get("longName") or ""
        except Exception:
            name = ""
        selected.append({
            "ticker": tk,
            "company": name,
            "daily_rsi": round(d, 2),
            "weekly_rsi": round(w, 2),
            "volatility": float(vol),
            "series_daily": daily.tail(40).tolist()
        })
    df = pd.DataFrame(selected).sort_values("volatility").reset_index(drop=True).head(N_CARDS)
    if df.empty:
        return df
    # signals
    def between(x): return 40 <= x <= 60
    labels, colors, crosses = [], [], []
    for _, row in df.iterrows():
        d = row["daily_rsi"]; w = row["weekly_rsi"]
        if (d > 60) and between(w):
            labels.append("Conflict"); colors.append(PALETTE["gray"]); crosses.append(True); continue
        if (w > 60) and between(d):
            labels.append("Conflict"); colors.append(PALETTE["gray"]); crosses.append(True); continue
        if (d > 60) and (w > 60):
            labels.append("Buy"); colors.append(PALETTE["green"]); crosses.append(False); continue
        if (d < 40) and (w < 40):
            labels.append("Sell"); colors.append(PALETTE["red"]); crosses.append(False); continue
        if between(d) and between(w):
            labels.append("Neutral"); colors.append(PALETTE["yellow"]); crosses.append(False); continue
        labels.append("Neutral"); colors.append(PALETTE["yellow"]); crosses.append(False)
    df["signal"] = labels
    df["color"] = colors
    df["cross"] = crosses
    return df

# -------------------------
# sparkline
# -------------------------
def sparkline_svg(values, width=140, height=36, stroke="#0F172A"):
    if not values or len(values) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'
    vals = np.array(values, dtype=float)
    minv, maxv = vals.min(), vals.max()
    if maxv == minv:
        ys = np.ones_like(vals) * (height / 2)
    else:
        ys = height - ((vals - minv) / (maxv - minv) * (height - 6)) - 3
    xs = np.linspace(2, width - 2, len(vals))
    path = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in zip(xs, ys))
    area = path + f" L {width-2} {height-2} L 2 {height-2} Z"
    svg = (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="{area}" fill="rgba(43,138,235,0.08)"></path>'
        f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>'
        '</svg>'
    )
    return svg

# -------------------------
# UI: detect refresh query param -> clear cache and remove it
# -------------------------
def handle_query_params():
    params = st.query_params      # modern replacement for experimental_get_query_params
    if "refresh" in params:
        # clear cache silently
        try:
            st.cache_data.clear()
        except Exception:
            pass

        # remove ?refresh param
        st.query_params = {}      # modern replacement for experimental_set_query_params

        # re-run safely
        try:
            st.experimental_rerun()   # still allowed as of 2025
        except Exception:
            pass


# -------------------------
# Build cards HTML and the centered HTML refresh button
# -------------------------
def build_page_html(df):
    font = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">'
    # center heading + HTML Refresh button
    # the button is an <a> that navigates top-level to ?refresh=<timestamp>, which reloads the app
    ts = int(time.time())
    refresh_link = f'?refresh={ts}'
    header_html = f"""
    <div style="text-align:center;margin:26px 0 12px 0;">
      <h1 style="font-family:Inter,system-ui; font-size:40px; font-weight:800; margin:0; color:{PALETTE['text']};">
        Least-Volatile NIFTY50 — Daily & Weekly RSI
      </h1>
      <div style="height:12px"></div>
      <a href="{refresh_link}" target="_top" style="text-decoration:none;">
        <button style="background:#fff;border:1px solid #E6EEF8;padding:8px 14px;border-radius:999px;box-shadow:0 6px 14px rgba(43,138,235,0.06);cursor:pointer;">
          🔄 Refresh
        </button>
      </a>
    </div>
    """

    # CSS for grid + cards; responsive 1/2/3 columns
    css = f"""
    <style>
      :root{{--text:{PALETTE['text']}; --muted:#6B7280; --shadow:{PALETTE['shadow']};}}
      body{{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:0; padding:0}}
      /* hide Streamlit's sidebar (extra safety) */
      section[data-testid="stSidebar"]{{display:none !important}}
      /* hide default footer/menu */
      footer{{visibility:hidden !important}}
      #MainMenu{{visibility:hidden !important}}
      .container{{max-width:1200px;margin:0 auto;padding:0 18px 28px;}}
      .grid{{ display:grid; gap:18px; grid-template-columns: repeat(1, minmax(0, 1fr)); }}
      @media(min-width:700px){{ .grid{{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
      @media(min-width:1100px){{ .grid{{ grid-template-columns: repeat(3, minmax(0,1fr)); }} }}

      .card{{
        display:flex;
        background: #fff;
        border-radius:12px;
        overflow:hidden;
        box-shadow: 0 10px 18px var(--shadow);
        transition: transform .16s ease, box-shadow .16s ease;
        min-height:120px;
      }}
      .card:hover{{ transform: translateY(-6px); box-shadow: 0 20px 36px rgba(2,6,23,0.12); }}
      .left-bar{{ width:10px; flex:0 0 10px; }}
      .card-body{{ padding:14px 16px; display:flex; flex-direction:column; gap:10px; flex:1; }}
      .ticker{{ font-weight:800; font-size:15px; color:var(--text); }}
      .company{{ font-size:12px; color:var(--muted); margin-top:2px; }}
      .rsi-row{{ display:flex; align-items:center; gap:18px; }}
      .rsi{{
        min-width:84px;
      }}
      .rsi-label{{ font-size:12px; color:#6b7280; }}
      .rsi-value{{ font-weight:800; font-size:15px; margin-top:4px; }}
      .signal{{ margin-top:8px; font-weight:700; font-size:13px; color:var(--muted); display:inline-block; }}
      .spark{{ flex:1; }}
    </style>
    """

    # build cards
    cards_html = ""
    for _, row in df.iterrows():
        color = row["color"]
        # left bar color darker shade for visibility
        left_bar = color
        # text color for values: dark for yellow background, dark text; otherwise dark text too for readability
        value_color = "#064E3B" if color != PALETTE["yellow"] else "#064E3B"
        spark = sparkline_svg(row.get("series_daily", [])[-30:], width=180, height=36, stroke="#0F172A")
        cross_html = '<div style="opacity:0.6;font-weight:800;">✖</div>' if row.get("cross") else ""
        company_html = f'<div class="company">{row["company"]}</div>' if row.get("company") else ''
        card = f"""
        <article class="card" role="article" aria-label="{row['ticker']}">
          <div class="left-bar" style="background:{left_bar};"></div>
          <div class="card-body">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
              <div>
                <div class="ticker">{row['ticker']}</div>
                {company_html}
              </div>
              <div style="margin-left:12px">{cross_html}</div>
            </div>

            <div class="rsi-row">
              <div class="rsi">
                <div class="rsi-label">Daily</div>
                <div class="rsi-value" style="color:{value_color};">{row['daily_rsi']:.2f}</div>
              </div>
              <div class="rsi">
                <div class="rsi-label">Weekly</div>
                <div class="rsi-value" style="color:{value_color};">{row['weekly_rsi']:.2f}</div>
              </div>
              <div class="spark">{spark}</div>
            </div>

            <div class="signal">{row['signal']}</div>
          </div>
        </article>
        """
        cards_html += card

    body = f'<div class="container">{header_html}<div class="grid">{cards_html}</div></div>'
    return font + css + body

# -------------------------
# Layout & run
# -------------------------
st.set_page_config(layout="wide", page_title="Least-Vol NIFTY50 RSI")
# extra CSS to hide streamlit padding (full-width feel)
st.markdown("<style>div.block-container{{padding-top:0rem;padding-left:0rem;padding-right:0rem;padding-bottom:0rem}}</style>", unsafe_allow_html=True)

# If user clicked the HTML refresh link, clear cache then remove param and rerun
handle_query_params()

with st.spinner("Fetching data and computing RSI..."):
    df = analyze_universe()

if df.empty:
    st.warning("No tickers available — possibly yfinance rate-limit or missing history. Try Refresh in a moment.")
    st.stop()

# build the page HTML
html = build_page_html(df)

# Compute a safe height so the HTML iframe is tall enough and doesn't clip:
# rows = ceil(n / columns), columns adapt by CSS but we choose 3 for height calc (safe overestimate)
cols_for_calc = 3
rows = math.ceil(len(df) / cols_for_calc)
iframe_height = max(360, rows * CARD_HEIGHT + 120)  # minimal safe height

# Render the whole page area in one component (scrolling allowed if window smaller)
components.html(html, height=iframe_height, scrolling=True)
