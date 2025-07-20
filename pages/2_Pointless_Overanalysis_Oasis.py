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
    
    # Mode selection in sidebar
    st.sidebar.header("Data Source")
    mode = st.sidebar.radio(
        "Select Mode",
        ["Season Mode", "Tournament Mode"],
        index=0,
        help="Season Mode uses regular match data, Tournament Mode uses tournament-specific data"
    )
    
    # Determine worksheet name based on mode
    if mode == "Tournament Mode":
        worksheet_name = "match_results_tournament"
    else:
        worksheet_name = "match_results"
    
    df = conn.read(worksheet=worksheet_name)

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

    # Date Range Filter
    min_date = df["date"].min()
    max_date = df["date"].max()
    start_date, end_date = st.sidebar.date_input(
        "Select date range to include",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )
    # Ensure they are Timestamps
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)

    # Day-of-Week Filter
    days_present_in_data = sorted(set(df["day_of_week"]))
    selected_days = st.sidebar.multiselect(
        "Select Day(s) of the Week to Include",
        options=days_present_in_data,
        default=days_present_in_data,
    )

    # Player Filter
    all_players = sorted(set(df["Player1"]) | set(df["Player2"]))
    selected_players = st.sidebar.multiselect(
        "Select Player(s) to Include", options=all_players, default=all_players
    )

    # Apply All Filters
    df_filtered = df[
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
        & (df["day_of_week"].isin(selected_days))
        & (df["Player1"].isin(selected_players))
        & (df["Player2"].isin(selected_players))
    ].copy()

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
