import streamlit as st
import pandas as pd
from display_utils import generate_analysis_content, display_enhanced_player_analysis
from streamlit_gsheets import GSheetsConnection


def main():
    # ------------- SETUP -------------
    st.set_page_config(page_title="Pointless Overanalysis Oasis", layout="wide", page_icon="📊")
    
    st.title("🏆 Pointless Overanalysis Oasis")
    st.markdown("*Delving Deep into Data Details - Now with Advanced Analytics!*")
    
    conn = st.connection("gsheets", type=GSheetsConnection)
    
    # Session selection in sidebar
    st.sidebar.header("Session Management")
    
    # Get current session from session state
    current_session = st.session_state.get("current_session", "")
    
    # If no session is selected, try to load the active session from Google Sheets
    if not current_session:
        try:
            sessions_df = conn.read(worksheet="sessions")
            if not sessions_df.empty and "session_name" in sessions_df.columns:
                # Check if there's an active session
                if "active" in sessions_df.columns:
                    active_sessions = sessions_df[sessions_df["active"] == True]
                    if not active_sessions.empty:
                        current_session = active_sessions.iloc[0]["session_name"]
                        st.session_state["current_session"] = current_session
                        st.sidebar.success(f"✅ Loaded active session: **{current_session}**")
                    else:
                        # No active session set, auto-select first available
                        available_sessions = sessions_df["session_name"].dropna().tolist()
                        if available_sessions:
                            current_session = available_sessions[0]
                            st.session_state["current_session"] = current_session
                            st.sidebar.info(f"🔄 Auto-selected session: **{current_session}**")
                            st.sidebar.info("🔧 Go to Settings to set an active session for all devices")
                        else:
                            st.sidebar.error("❌ No sessions found!")
                            st.sidebar.markdown("**Please go to Settings to create a session:**")
                            if st.sidebar.button("🔧 Go to Settings"):
                                st.switch_page("pages/3_Settings.py")
                            st.stop()
                else:
                    # No active column, auto-select first available
                    available_sessions = sessions_df["session_name"].dropna().tolist()
                    if available_sessions:
                        current_session = available_sessions[0]
                        st.session_state["current_session"] = current_session
                        st.sidebar.info(f"🔄 Auto-selected session: **{current_session}**")
                        st.sidebar.info("🔧 Go to Settings to set an active session for all devices")
                    else:
                        st.sidebar.error("❌ No sessions found!")
                        st.sidebar.markdown("**Please go to Settings to create a session:**")
                        if st.sidebar.button("🔧 Go to Settings"):
                            st.switch_page("pages/3_Settings.py")
                        st.stop()
            else:
                # Sessions worksheet exists but has no data
                st.sidebar.error("❌ No sessions found!")
                st.sidebar.markdown("**Please go to Settings to create a session:**")
                if st.sidebar.button("🔧 Go to Settings"):
                    st.switch_page("pages/3_Settings.py")
                st.stop()
        except Exception:
            # Sessions worksheet doesn't exist
            st.sidebar.error("❌ No sessions found!")
            st.sidebar.markdown("**Please go to Settings to create your first session:**")
            if st.sidebar.button("🔧 Go to Settings"):
                st.switch_page("pages/3_Settings.py")
            st.stop()
    
    # Show current session status
    st.sidebar.success(f"✅ Active Session: **{current_session}**")
    if st.sidebar.button("🔧 Change Session"):
        st.switch_page("pages/3_Settings.py")
    
    # Determine worksheet name based on current session
    worksheet_name = f"{current_session}_match_results"
    
    try:
        df = conn.read(worksheet=worksheet_name)
        
        # Early data validation
        if df.empty:
            st.warning(f"📋 Session worksheet '{worksheet_name}' is empty or doesn't exist yet.")
            st.info("🎾 Please go to the **Pointless Racquet Records** page to enter some match results first!")
            st.stop()
                
    except Exception as e:
        st.warning(f"📋 Session worksheet '{worksheet_name}' doesn't exist yet.")
        st.info("🎾 Please go to the **Pointless Racquet Records** page to enter some match results first!")
        st.stop()

    # ------------- DATA PREPROCESSING -------------
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # Some name corrections if needed
    df["Player1"] = df["Player1"].replace("Friede", "Friedemann")
    df["Player2"] = df["Player2"].replace("Friede", "Friedemann")

    # Derive winner/loser and related columns
    df["Winner"] = df.apply(
        lambda row: row["Player1"] if row["Score1"] > row["Score2"] else row["Player2"],
        axis=1,
    )
    df["Loser"] = df.apply(
        lambda row: row["Player2"] if row["Score1"] > row["Score2"] else row["Player1"],
        axis=1,
    )
    df["WinnerScore"] = df[["Score1", "Score2"]].max(axis=1)
    df["LoserScore"] = df[["Score1", "Score2"]].min(axis=1)
    df["PointDiff"] = df["WinnerScore"] - df["LoserScore"]
    df["LoserPointDiff"] = df["LoserScore"] - df["WinnerScore"]
    # day_of_week column
    df["day_of_week"] = df["date"].dt.day_name()

    # ------------- SIDEBAR FILTERS -------------
    st.sidebar.header("Filters")

    # Date Range Filter - Handle cases where dates might be invalid
    valid_dates = df["date"].dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        start_date, end_date = st.sidebar.date_input(
            "Select date range to include",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
        )
    else:
        # Fallback to default dates if no valid dates exist
        import datetime
        default_date = datetime.date.today()
        st.sidebar.warning("No valid dates found in data. Using default date range.")
        start_date, end_date = st.sidebar.date_input(
            "Select date range to include",
            [default_date, default_date],
            min_value=default_date,
            max_value=default_date,
        )
    # Ensure they are Timestamps
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)

    # Day-of-Week Filter - Handle missing values
    days_present_in_data = sorted(set(df["day_of_week"].dropna()))
    if len(days_present_in_data) > 0:
        selected_days = st.sidebar.multiselect(
            "Select Day(s) of the Week to Include",
            options=days_present_in_data,
            default=days_present_in_data,
        )
    else:
        st.sidebar.warning("No valid days of week found in data.")
        selected_days = []

    # Player Filter - Handle missing values
    all_players = sorted((set(df["Player1"].dropna()) | set(df["Player2"].dropna())))
    if len(all_players) > 0:
        selected_players = st.sidebar.multiselect(
            "Select Player(s) to Include", options=all_players, default=all_players
        )
    else:
        st.sidebar.warning("No valid players found in data.")
        selected_players = []

    # Apply All Filters - with error handling
    try:
        df_filtered = df[
            (df["date"] >= start_date)
            & (df["date"] <= end_date)
            & (df["day_of_week"].isin(selected_days))
            & (df["Player1"].isin(selected_players))
            & (df["Player2"].isin(selected_players))
        ].copy()
        
        # Check if filtered dataframe is empty
        if df_filtered.empty:
            st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        st.warning("Using unfiltered data due to filtering error.")
        df_filtered = df.copy()

    # ------------- MAIN TABS -------------
    main_tab_overall, main_tab_player = st.tabs(
        ["Overall Overanalysis", "Player Analysis"]
    )

    # Overall Analysis Tab
    with main_tab_overall:
        generate_analysis_content(df_filtered, include_ratings=True)

    # Player Analysis Tab (Enhanced)
    with main_tab_player:
        display_enhanced_player_analysis(df_filtered)


# Run the app
if __name__ == "__main__":
    main()
