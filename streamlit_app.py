import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import os
import pytz
from streamlit_js_eval import streamlit_js_eval
from lightweight_charts.widgets import StreamlitChart
from libsql_client import create_client_sync, ClientSync

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    layout="wide",
    page_title="Market Rewind",
    initial_sidebar_state="collapsed"
)

# ========================================
# CSS: CLEAN PADDING
# ========================================
st.markdown("""
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
""", unsafe_allow_html=True)

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
        # Construct query based on ETH toggle
        # If ETH is True, we remove the session filter to get ALL sessions (REG + EXT)
        # If ETH is False, we restrict to 'REG' only
        
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
    df['time'] = pd.to_datetime(df['timestamp'], utc=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    # Color logic
    df['color'] = np.where(df['open'] > df['close'],
                           'rgba(239, 83, 80, 0.8)',
                           'rgba(38, 166, 154, 0.8)')
    
    return df[['time', 'open', 'high', 'low', 'close', 'volume', 'color']]

@st.cache_data
def resample_data(df, timeframe):
    if df.empty:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'color'])

    df = df.set_index('time')
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    resampled = df.resample(timeframe).agg(agg).dropna().reset_index()
    
    # Re-apply color after resampling
    resampled['color'] = np.where(resampled['open'] > resampled['close'],
                                  'rgba(239, 83, 80, 0.8)',
                                  'rgba(38, 166, 154, 0.8)')
    return resampled

# ========================================
# CHART UNIT
# ========================================

# --- FIX: @st.fragment DECORATOR ADDED HERE ---
# This isolates the rerun loop to ONLY this specific chart unit function.
# The rest of the page (Layout controls, Headers) will NOT reload during playback.
@st.fragment
def render_chart_unit(chart_id, db_client, show_border=True, default_tf="1 Min"):
    """
    Renders a completely independent chart unit with automatic replay loading.
    """
    # STATE KEYS
    k_ticker = f"c{chart_id}_ticker"
    k_tf = f"c{chart_id}_tf"
    k_eth = f"c{chart_id}_eth" 
    k_view_mode = f"c{chart_id}_view_mode" 
    k_active = f"c{chart_id}_active"
    k_paused = f"c{chart_id}_paused"
    k_index = f"c{chart_id}_index"
    k_data = f"c{chart_id}_data"
    k_date = f"c{chart_id}_date"
    k_speed = f"c{chart_id}_speed"
    k_last_sig = f"c{chart_id}_last_signature" 
    k_prev_date = f"c{chart_id}_prev_date" # Fixed: Added missing definition for date tracking key

    # Initialize State
    if k_active not in st.session_state:
        st.session_state[k_active] = True 
        st.session_state[k_paused] = True
        st.session_state[k_index] = 0
        st.session_state[k_data] = pd.DataFrame()
    
    if k_last_sig not in st.session_state:
        st.session_state[k_last_sig] = None

    if k_prev_date not in st.session_state:
        st.session_state[k_prev_date] = None

    if k_tf not in st.session_state:
        st.session_state[k_tf] = default_tf
        
    if k_eth not in st.session_state:
        st.session_state[k_eth] = False 

    # New State for View Mode (Selector)
    if k_view_mode not in st.session_state:
        st.session_state[k_view_mode] = "Viewer Mode"

    chart_height = st.session_state.get("chart_height_px", 600)

    with st.container(border=show_border):

        # --- CONTROLS ROW 1: SELECTORS ---
        c1, c2, c3, c4, _ = st.columns([1.5, 1.5, 2.0, 1.0, 1.0])
        
        with c1:
            tickers = get_available_tickers(db_client)
            sel_ticker = st.selectbox(
                "Ticker", tickers,
                key=k_ticker,
                label_visibility="collapsed",
                placeholder="Ticker",
                help="Select the stock symbol to analyze."
            )
        with c2:
            tf_map = {"1 Min": "1min", "5 Min": "5min", "15 Min": "15min", "30 Min": "30min", "1 Hr": "1H", "1 Day": "1D"}
            sel_tf_str = st.selectbox(
                "TF", list(tf_map.keys()), 
                key=k_tf, 
                label_visibility="collapsed",
                help="Select the chart timeframe."
            )
            sel_tf_agg = tf_map[sel_tf_str]
            
        with c3:
            # View Mode Selector (Replaces Toggle)
            st.selectbox(
                "Mode", 
                ["Viewer Mode", "Replay Mode"], 
                key=k_view_mode, 
                label_visibility="collapsed",
                help="Viewer Mode: Dense, thin candles for overview.\nReplay Mode: Zoomed, thick candles for simulation."
            )

        with c4:
            # ETH Toggle Button (Moved to end)
            is_eth = st.toggle(
                "ETH", 
                key=k_eth,
                help="Toggle Extended Trading Hours (Pre/Post Market).\n\n⚠️ **WARNING:** Toggling this will reset the chart and lost replay progress!"
            )

        # Logic check for mode
        is_replay_mode = (st.session_state[k_view_mode] == "Replay Mode")

        # --- DATA PREP (AUTO-LOAD LOGIC) ---
        if not sel_ticker:
            st.info("Select ticker")
            return

        EARLIEST = "2024-01-01"
        master_data = load_master_data(db_client, sel_ticker, EARLIEST, is_eth)
        
        if master_data.empty:
            st.warning("No data found.")
            return
        
        # --- SMART DATE LOGIC ---
        # 1. Find latest available date in the DB for this ticker (for initial default only)
        latest_db_date = master_data['time'].max().date()
        
        # 2. Only set the date to the latest DB date if it hasn't been set yet.
        #    Once set, we DO NOT overwrite it when the ticker changes.
        if k_date not in st.session_state:
            st.session_state[k_date] = latest_db_date

        current_date_val = st.session_state[k_date]
        # ------------------------

        current_signature = f"{sel_ticker}_{sel_tf_agg}_{current_date_val}_{is_eth}"
        
        if st.session_state[k_last_sig] != current_signature:
            # === RECALCULATE STATE ===
            
            # A. Resample Full History
            full_resampled = resample_data(master_data, sel_tf_agg)
            
            # B. Calculate Start Index
            if not full_resampled.empty:
                # Determine the target time to sync to
                
                # Default target: 9:30 AM ET on the selected date
                ny_tz = pytz.timezone('America/New_York')
                start_dt_ny = datetime.datetime.combine(current_date_val, datetime.time(9, 30))
                target_dt_aware = ny_tz.localize(start_dt_ny).astimezone(pytz.UTC)

                # SYNC LOGIC: If date hasn't changed (meaning Ticker/TF changed), try to keep current replay time
                prev_date_val = st.session_state.get(k_prev_date)
                
                # Check if we have existing data to grab time from
                if prev_date_val == current_date_val and not st.session_state[k_data].empty:
                    try:
                        # Get current replay time from old data before we overwrite it
                        curr_idx = min(st.session_state[k_index], len(st.session_state[k_data]) - 1)
                        current_replay_time = st.session_state[k_data].iloc[curr_idx]['time']
                        # Update target to current replay time
                        target_dt_aware = current_replay_time
                    except Exception:
                        pass # Fallback to 9:30 AM if error

                # Find index for target time in NEW data
                start_index = full_resampled['time'].searchsorted(target_dt_aware)
                start_index = min(max(0, start_index), len(full_resampled) - 1)
                
                # Update State with new data and synced index
                st.session_state[k_data] = full_resampled
                st.session_state[k_index] = int(start_index)
            else:
                 st.session_state[k_data] = pd.DataFrame()
                 st.session_state[k_index] = 0
            
            st.session_state[k_paused] = True
            st.session_state[k_active] = True 
            st.session_state[k_last_sig] = current_signature
            st.session_state[k_prev_date] = current_date_val


        # --- CHART RENDERING ---
        try:
            chart = StreamlitChart(height=chart_height)
            chart.layout(background_color="#0f111a", text_color="#ffffff")
            chart.price_scale()
            chart.volume_config()

            # --- DYNAMIC VISUAL SETTINGS ---
            if is_replay_mode:
                # REPLAY MODE: Very thick bars, Wide offset
                offset = 45 # Increased for more y-axis distance
                
                if sel_tf_str == "1 Min": spacing = 8.0 # Thicker 1-min
                elif sel_tf_str == "5 Min": spacing = 10.0
                elif sel_tf_str == "15 Min": spacing = 12.0
                elif sel_tf_str == "30 Min": spacing = 14.0
                elif sel_tf_str == "1 Hr": spacing = 16.0
                elif sel_tf_str == "1 Day": spacing = 20.0
                else: spacing = 10.0
            else:
                # VIEWER MODE: Thin bars, Tight offset (Overview)
                offset = 5
                if sel_tf_str == "1 Min": spacing = 0.5
                elif sel_tf_str == "5 Min": spacing = 2.0
                elif sel_tf_str == "15 Min": spacing = 4.0
                elif sel_tf_str == "30 Min": spacing = 7.0
                elif sel_tf_str == "1 Hr": spacing = 8.0
                elif sel_tf_str == "1 Day": spacing = 10.0
                else: spacing = 5.0

            chart.time_scale(min_bar_spacing=spacing, right_offset=offset)

        except Exception as e:
            st.error(f"Chart Error: {e}")
            return

        if not st.session_state[k_data].empty:
            current_slice = st.session_state[k_data].iloc[:st.session_state[k_index]]
            if not current_slice.empty:
                c_data = current_slice.copy()
                c_data['time'] = c_data['time'].apply(lambda x: x.isoformat())
                chart.set(c_data)
            else:
                 chart.set(pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'color']))
            
            # Helper: Replay Progress Text (helps verify speed)
            if is_replay_mode:
                try:
                    last_time = current_slice['time'].iloc[-1]
                    # Format timestamp nicely
                    ts_str = last_time.strftime('%H:%M')
                except:
                    ts_str = "--:--"
                st.caption(f"Replay: {ts_str} ({st.session_state[k_index]}/{len(st.session_state[k_data])})")

        else:
            chart.set(pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'color']))

        chart.load()

        # --- BOTTOM CONTROLS ---
        # Revised Column Distribution for New Buttons
        # Date | Prev | Play | Next | Reset | Speed
        c_date, c_prev, c_play, c_next, c_reset, c_speed = st.columns([2, 0.7, 1.5, 0.7, 1.5, 1.5])

        with c_date:
            st.date_input(
                "Start", 
                # Removed value argument to fix Streamlit error
                key=k_date, 
                label_visibility="collapsed",
                help="Select the starting date for the replay."
            )
        
        with c_prev:
            if st.button("⏮", key=f"btn_prev_{chart_id}", use_container_width=True, help="Step Back (Previous Candle)"):
                if st.session_state[k_index] > 0:
                    st.session_state[k_index] -= 1
                    st.rerun()

        with c_play:
            if st.session_state[k_paused]:
                if st.button("▶ Play", key=f"btn_play_{chart_id}", use_container_width=True):
                    st.session_state[k_paused] = False
                    st.rerun()
            else:
                if st.button("⏸ Pause", key=f"btn_pause_{chart_id}", use_container_width=True):
                    st.session_state[k_paused] = True
                    st.rerun()
        
        with c_next:
            if st.button("⏭", key=f"btn_next_{chart_id}", use_container_width=True, help="Step Forward (Next Candle)"):
                if st.session_state[k_index] < len(st.session_state[k_data]):
                    st.session_state[k_index] += 1
                    st.rerun()

        with c_reset:
            if st.button("↺ Reset", key=f"btn_reset_{chart_id}", use_container_width=True):
                st.session_state[k_last_sig] = None 
                st.rerun()

        with c_speed:
            # Sort speed options for better UX
            speed_options = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
            r_speed = st.selectbox(
                "Spd", 
                speed_options,
                index=3, # Default to 1.0s
                format_func=lambda x: f"{x}s",
                key=k_speed,
                label_visibility="collapsed",
                help="Time delay between each candle update (lower is faster)."
            )

        # --- PLAY LOOP ---
        if not st.session_state[k_paused]:
            if st.session_state[k_index] < len(st.session_state[k_data]):
                st.session_state[k_index] += 1
                # Explicit sleep with float conversion for safety
                time.sleep(float(r_speed))
                st.rerun()
            else:
                st.session_state[k_paused] = True
                st.rerun()

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

    # ===== GLOBAL CHART CONTROLS =====
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

    # ====== RENDER GRID ======
    if n == 1:
        render_chart_unit(0, db_client, show_border=False)

    elif n == 2:
        c1, c2 = st.columns(2)
        with c1: render_chart_unit(0, db_client)
        with c2: render_chart_unit(1, db_client)

    elif n == 3:
        c1, c2 = st.columns(2)
        with c1: render_chart_unit(0, db_client)
        with c2: render_chart_unit(1, db_client)
        render_chart_unit(2, db_client)

    elif n == 4:
        c1, c2 = st.columns(2)
        with c1: render_chart_unit(0, db_client)
        with c2: render_chart_unit(1, db_client)

        c3, c4 = st.columns(2)
        with c3: render_chart_unit(2, db_client)
        with c4: render_chart_unit(3, db_client)