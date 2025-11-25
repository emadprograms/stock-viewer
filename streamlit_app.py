import os
import time
import datetime

import numpy as np
import pandas as pd
import pytz
import streamlit as st
from libsql_client import ClientSync, create_client_sync
from lightweight_charts.widgets import StreamlitChart
from streamlit_autorefresh import st_autorefresh  # requires: pip install streamlit-autorefresh
from streamlit_js_eval import streamlit_js_eval

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    layout="wide",
    page_title="Market Rewind",
    initial_sidebar_state="collapsed",
)

# ========================================
# CSS: CLEAN PADDING
# ========================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.5rem;
        }
        /* Better alignment for the playback buttons */
        div.stButton > button {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================================
# DYNAMIC HEIGHT CALC
# ========================================
def get_dynamic_chart_height(num_charts, viewport_height):
    """
    Compute per-chart height based on viewport.
    """
    if viewport_height is None or viewport_height <= 0:
        viewport_height = 900

    reserved_top_px = 40
    reserved_bottom_px = 10

    usable = max(300, viewport_height - reserved_top_px - reserved_bottom_px)

    rows = 1 if num_charts <= 2 else 2
    per_chart = usable / rows

    return int(per_chart)


# ========================================
# DATABASE CONNECTION
# ========================================
@st.cache_resource
def get_db_connection():
    try:
        if "turso" in st.secrets:
            url = st.secrets["turso"]["db_url"]
            token = st.secrets["turso"]["auth_token"]
        else:
            url = os.environ.get("TURSO_DB_URL")
            token = os.environ.get("TURSO_AUTH_TOKEN")

        if not url or not token:
            st.error("Missing Turso credentials.")
            return None

        http_url = url.replace("libsql://", "https://")
        config = {"url": http_url, "auth_token": token}
        return create_client_sync(**config)
    except Exception as e:
        st.error(f"Failed to create Turso client: {e}")
        return None


@st.cache_data
def get_available_tickers(_client: ClientSync):
    try:
        rs = _client.execute("SELECT user_ticker FROM symbol_map ORDER BY user_ticker;")
        return [row["user_ticker"] for row in rs.rows]
    except Exception as e:
        st.error(f"Failed to fetch tickers: {e}")
        return []


@st.cache_data
def load_master_data(_client: ClientSync, ticker: str, earliest_date_str: str, include_eth: bool):
    try:
        if include_eth:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp;
            """
            params = [ticker, earliest_date_str]
        else:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND session = 'REG' AND timestamp >= ?
                ORDER BY timestamp;
            """
            params = [ticker, earliest_date_str]

        rs = _client.execute(query, params)

    except Exception as e:
        st.error(f"DB Query failed for {ticker}: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(rs.rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df

    # Ensure timezone aware (UTC)
    df["time"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    # Color logic
    df["color"] = np.where(
        df["open"] > df["close"],
        "rgba(239, 83, 80, 0.8)",
        "rgba(38, 166, 154, 0.8)",
    )

    return df[["time", "open", "high", "low", "close", "volume", "color"]]


@st.cache_data
def resample_data(df, timeframe):
    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "color"])

    df = df.set_index("time")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = df.resample(timeframe).agg(agg).dropna().reset_index()

    # Re-apply color after resampling
    resampled["color"] = np.where(
        resampled["open"] > resampled["close"],
        "rgba(239, 83, 80, 0.8)",
        "rgba(38, 166, 154, 0.8)",
    )
    return resampled


# ========================================
# GLOBAL TIMEFRAME MAP & PLAYBACK HELPERS
# ========================================
TIMEFRAME_MAP = {
    "1 Min": ("1min", 1),
    "5 Min": ("5min", 5),
    "15 Min": ("15min", 15),
    "30 Min": ("30min", 30),
    "1 Hr": ("1H", 60),
    "1 Day": ("1D", 1440),
}


def init_global_playback_state():
    if "global_playing" not in st.session_state:
        st.session_state["global_playing"] = False
    if "global_speed" not in st.session_state:
        st.session_state["global_speed"] = 1.0  # seconds
    if "global_date" not in st.session_state:
        st.session_state["global_date"] = None
    if "global_dt" not in st.session_state:
        st.session_state["global_dt"] = None
    if "global_date_prev" not in st.session_state:
        st.session_state["global_date_prev"] = None
    if "global_last_tick" not in st.session_state:
        st.session_state["global_last_tick"] = None


def get_min_step_minutes(num_charts: int) -> int:
    mins = []
    for i in range(num_charts):
        tf_label = st.session_state.get(f"c{i}_tf", "1 Min")
        _, step_min = TIMEFRAME_MAP.get(tf_label, ("1min", 1))
        mins.append(step_min)
    return min(mins) if mins else 1


def ensure_global_dt_initialized():
    """Ensure global_dt exists, using global_date at 9:30 AM ET."""
    if st.session_state.get("global_date") is None:
        return
    if st.session_state.get("global_dt") is None:
        ny_tz = pytz.timezone("America/New_York")
        start_dt_ny = datetime.datetime.combine(
            st.session_state["global_date"],
            datetime.time(9, 30),
        )
        st.session_state["global_dt"] = ny_tz.localize(start_dt_ny).astimezone(pytz.UTC)


def step_global_dt(direction: int = 1):
    """Move global_dt by +/- one smallest timeframe unit."""
    ensure_global_dt_initialized()
    if st.session_state.get("global_dt") is None:
        return
    n = st.session_state.get("num_charts", 1)
    minutes = get_min_step_minutes(n)
    delta = datetime.timedelta(minutes=minutes * direction)
    st.session_state["global_dt"] = st.session_state["global_dt"] + delta


def render_global_controls():
    init_global_playback_state()
    ensure_global_dt_initialized()

    c_date, c_prev, c_play, c_next, c_speed = st.columns([2, 0.7, 1.5, 0.7, 1.5])

    # Global Date picker: reset global_dt to 9:30 AM ET on new date
    with c_date:
        date_val = st.date_input(
            "Date",
            key="global_date",
            help="Trading date for Market Rewind (resets time to 9:30 AM ET when changed).",
        )
        if st.session_state["global_date_prev"] != date_val:
            st.session_state["global_date_prev"] = date_val
            st.session_state["global_date"] = date_val
            ny_tz = pytz.timezone("America/New_York")
            start_dt_ny = datetime.datetime.combine(
                date_val,
                datetime.time(9, 30),
            )
            st.session_state["global_dt"] = ny_tz.localize(start_dt_ny).astimezone(pytz.UTC)
            st.session_state["global_playing"] = False
            st.rerun()

    # Previous: step back by one unit
    with c_prev:
        if st.button("⏮", use_container_width=True, help="Step Back (Previous Unit)"):
            step_global_dt(direction=-1)
            st.rerun()

    # Play / Pause toggle
    with c_play:
        if st.session_state.get("global_playing", False):
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state["global_playing"] = False
                st.rerun()
        else:
            if st.button("▶ Play", use_container_width=True):
                st.session_state["global_playing"] = True
                st.rerun()

    # Next: step forward by one unit
    with c_next:
        if st.button("⏭", use_container_width=True, help="Step Forward (Next Unit)"):
            step_global_dt(direction=1)
            st.rerun()

    # Global speed selector
    with c_speed:
        speed_options = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        current_speed = st.session_state.get("global_speed", 1.0)
        if current_speed not in speed_options:
            current_speed = 1.0
        st.session_state["global_speed"] = st.selectbox(
            "Spd",
            speed_options,
            index=speed_options.index(current_speed),
            format_func=lambda x: f"{x:.2f}s" if x < 1 else f"{x:.0f}s",
            key="global_speed_widget",
            label_visibility="collapsed",
            help="Time delay between each global step (lower is faster).",
        )

    # Display current global time in NY
    if st.session_state.get("global_dt") is not None:
        ny_tz = pytz.timezone("America/New_York")
        ts_ny = st.session_state["global_dt"].astimezone(ny_tz)
        st.caption(f"Global replay time: {ts_ny.strftime('%Y-%m-%d %H:%M %Z')}")


# ========================================
# CHART UNIT (PURE VIEW OF GLOBAL_DT)
# ========================================
@st.fragment
def render_chart_unit(chart_id, db_client, show_border=True, default_tf="1 Min"):
    """
    Renders a chart unit that is fully driven by global_dt.
    No local playback loop or local time state.
    """
    # STATE KEYS
    k_ticker = f"c{chart_id}_ticker"
    k_tf = f"c{chart_id}_tf"
    k_eth = f"c{chart_id}_eth"
    k_view_mode = f"c{chart_id}_view_mode"
    k_active = f"c{chart_id}_active"
    k_data = f"c{chart_id}_data"
    k_last_sig = f"c{chart_id}_last_signature"

    # Initialize per-chart state
    if k_active not in st.session_state:
        st.session_state[k_active] = True
    if k_tf not in st.session_state:
        st.session_state[k_tf] = default_tf
    if k_eth not in st.session_state:
        st.session_state[k_eth] = False
    if k_view_mode not in st.session_state:
        st.session_state[k_view_mode] = "Viewer Mode"
    if k_data not in st.session_state:
        st.session_state[k_data] = pd.DataFrame()
    if k_last_sig not in st.session_state:
        st.session_state[k_last_sig] = None

    chart_height = st.session_state.get("chart_height_px", 600)

    with st.container(border=show_border):
        # --- CONTROLS ROW 1: SELECTORS ---
        c1, c2, c3, c4, _ = st.columns([1.5, 1.5, 2.0, 1.0, 1.0])

        with c1:
            tickers = get_available_tickers(db_client)
            sel_ticker = st.selectbox(
                "Ticker",
                tickers,
                key=k_ticker,
                label_visibility="collapsed",
                placeholder="Ticker",
                help="Select the stock symbol to analyze.",
            )

        with c2:
            tf_label = st.selectbox(
                "TF",
                list(TIMEFRAME_MAP.keys()),
                key=k_tf,
                label_visibility="collapsed",
                help="Select the chart timeframe.",
            )
            sel_tf_agg, _ = TIMEFRAME_MAP[tf_label]

        with c3:
            st.selectbox(
                "Mode",
                ["Viewer Mode", "Replay Mode"],
                key=k_view_mode,
                label_visibility="collapsed",
                help=(
                    "Viewer Mode: Dense, thin candles for overview.\n"
                    "Replay Mode: Zoomed, thick candles for simulation."
                ),
            )

        with c4:
            is_eth = st.toggle(
                "ETH",
                key=k_eth,
                help=(
                    "Toggle Extended Trading Hours (Pre/Post Market).\n\n"
                    "⚠️ Toggling this will reload data for this chart."
                ),
            )

        is_replay_mode = st.session_state[k_view_mode] == "Replay Mode"

        # --- DATA PREP ---
        if not sel_ticker:
            st.info("Select ticker")
            return

        EARLIEST = "2024-01-01"
        master_data = load_master_data(db_client, sel_ticker, EARLIEST, is_eth)
        if master_data.empty:
            st.warning("No data found.")
            return

        # Initialize global_date / global_dt once using actual data
        latest_db_date = master_data["time"].max().date()
        if st.session_state.get("global_date") is None:
            st.session_state["global_date"] = latest_db_date
        ensure_global_dt_initialized()

        # (Re)build resampled data only when signature changes
        current_signature = f"{sel_ticker}_{sel_tf_agg}_{is_eth}"
        if st.session_state[k_last_sig] != current_signature:
            full_resampled = resample_data(master_data, sel_tf_agg)
            st.session_state[k_data] = full_resampled
            st.session_state[k_last_sig] = current_signature

        df = st.session_state[k_data]

        # --- CHART RENDERING ---
        try:
            chart = StreamlitChart(height=chart_height)
            chart.layout(background_color="#0f111a", text_color="#ffffff")
            chart.price_scale()
            chart.volume_config()

            # DYNAMIC VISUAL SETTINGS
            if is_replay_mode:
                offset = 45
                if tf_label == "1 Min":
                    spacing = 8.0
                elif tf_label == "5 Min":
                    spacing = 10.0
                elif tf_label == "15 Min":
                    spacing = 12.0
                elif tf_label == "30 Min":
                    spacing = 14.0
                elif tf_label == "1 Hr":
                    spacing = 16.0
                elif tf_label == "1 Day":
                    spacing = 20.0
                else:
                    spacing = 10.0
            else:
                offset = 5
                if tf_label == "1 Min":
                    spacing = 0.5
                elif tf_label == "5 Min":
                    spacing = 2.0
                elif tf_label == "15 Min":
                    spacing = 4.0
                elif tf_label == "30 Min":
                    spacing = 7.0
                elif tf_label == "1 Hr":
                    spacing = 8.0
                elif tf_label == "1 Day":
                    spacing = 10.0
                else:
                    spacing = 5.0

            chart.time_scale(min_bar_spacing=spacing, right_offset=offset)

        except Exception as e:
            st.error(f"Chart Error: {e}")
            return

        # --- APPLY GLOBAL TIME TO THIS CHART ---
        if (
            not df.empty
            and st.session_state.get("global_dt") is not None
        ):
            global_dt = st.session_state["global_dt"]
            # include all candles <= global_dt
            idx = df["time"].searchsorted(global_dt, side="right")
            idx = min(max(0, idx), len(df))
            current_slice = df.iloc[:idx].copy()
        else:
            current_slice = pd.DataFrame(columns=df.columns) if not df.empty else df

        if not current_slice.empty:
            current_slice["time"] = current_slice["time"].apply(lambda x: x.isoformat())
            chart.set(current_slice)
            if is_replay_mode:
                # current_slice["time"] is ISO strings now; parse for display
                try:
                    last_time_dt = pd.to_datetime(current_slice["time"].iloc[-1])
                    ts_str = last_time_dt.strftime("%H:%M")
                except Exception:
                    ts_str = "--:--"
                st.caption(f"Replay: {ts_str} ({len(current_slice)}/{len(df)})")
        else:
            chart.set(
                pd.DataFrame(
                    columns=["time", "open", "high", "low", "close", "volume", "color"],
                )
            )

        chart.load()


# ========================================
# MAIN EXECUTION FLOW
# ========================================
db_client = get_db_connection()
if not db_client:
    st.stop()

st.markdown("### Market Rewind")

# STEP 1: LAYOUT CONFIG
if "layout_set" not in st.session_state:
    st.info("Configure your workspace to begin.")
    with st.form("layout_config"):
        num_charts = st.selectbox("How many charts do you want?", [1, 2, 3, 4])
        submitted = st.form_submit_button("Initialize Workspace")
        if submitted:
            st.session_state.num_charts = num_charts
            st.session_state.layout_set = True
            st.rerun()

else:
    n = st.session_state.num_charts

    # Initialize global playback state now that n is known
    init_global_playback_state()

    # ===== GLOBAL CHART CONTROLS (LAYOUT / HEIGHT) =====
    with st.expander("Global chart controls", expanded=False):
        screen_height = streamlit_js_eval(
            js_code="window.innerHeight",
            key="screen_height_js",
            default=1080,
        )
        default_height = int(screen_height or 1080)

        label_col_btn, label_col_height = st.columns(2)
        with label_col_btn:
            st.markdown("🔁 Layout & workspace")
        with label_col_height:
            st.markdown("↕ Height override (px)")

        col_btn, col_height = st.columns(2)
        with col_btn:
            if st.button(
                "Click here to reconfigure charts",
                type="secondary",
                use_container_width=True,
            ):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        with col_height:
            manual_height = st.number_input(
                "Height override (px)",
                min_value=600,
                max_value=2000,
                value=default_height,
                step=10,
                label_visibility="collapsed",
            )

    viewport_height = manual_height or screen_height
    st.session_state["chart_height_px"] = get_dynamic_chart_height(n, viewport_height)

    # ===== NON-BLOCKING GLOBAL PLAYBACK TIMER =====
    if st.session_state.get("global_playing", False):
        interval_ms = int(st.session_state.get("global_speed", 1.0) * 1000)
        tick = st_autorefresh(
            interval=interval_ms,
            limit=None,
            key="global_playback_tick",
            debounce=True,
        )
        last_tick = st.session_state.get("global_last_tick")
        if last_tick is None:
            st.session_state["global_last_tick"] = tick
        elif tick != last_tick:
            st.session_state["global_last_tick"] = tick
            step_global_dt(direction=1)

    # ====== RENDER GRID ======
    if n == 1:
        render_chart_unit(0, db_client, show_border=False)

    elif n == 2:
        c1, c2 = st.columns(2)
        with c1:
            render_chart_unit(0, db_client)
        with c2:
            render_chart_unit(1, db_client)

    elif n == 3:
        c1, c2 = st.columns(2)
        with c1:
            render_chart_unit(0, db_client)
        with c2:
            render_chart_unit(1, db_client)
        render_chart_unit(2, db_client)

    elif n == 4:
        c1, c2 = st.columns(2)
        with c1:
            render_chart_unit(0, db_client)
        with c2:
            render_chart_unit(1, db_client)

        c3, c4 = st.columns(2)
        with c3:
            render_chart_unit(2, db_client)
        with c4:
            render_chart_unit(3, db_client)

    # ===== GLOBAL CONTROL BAR (BOTTOM) =====
    st.markdown("---")
    render_global_controls()
