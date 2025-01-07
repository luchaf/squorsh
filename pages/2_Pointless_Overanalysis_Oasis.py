import streamlit as st
import pandas as pd
from display_utils import generate_analysis_content
from streamlit_gsheets import GSheetsConnection
from color_palette import PRIMARY, SECONDARY, TERTIARY


def main():
    # ------------- SETUP -------------
    st.set_page_config(layout="wide", primaryColor=PRIMARY)
    conn = st.connection("gsheets", type=GSheetsConnection)
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
    st.sidebar.header("Filters", color=PRIMARY)

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
    main_tab_overall, main_tab_head2head = st.tabs(
        ["Overall Overanalysis", "Head-to-Head Overanalysis"]
    )

    # Overall Analysis Tab
    with main_tab_overall:
        generate_analysis_content(df_filtered, include_ratings=True)

    # Head-to-Head Analysis Tab
    with main_tab_head2head:
        st.subheader("Select Players for Head-to-Head Analysis", color=SECONDARY)
        players = [""] + sorted(set(df["Player1"]) | set(df["Player2"]))

        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox(
                "Select Player 1",
                players,
                format_func=lambda x: "Select..." if x == "" else x,
            )
        with col2:
            player2 = st.selectbox(
                "Select Player 2",
                players,
                format_func=lambda x: "Select..." if x == "" else x,
            )

        if player1 and player2 and player1 != player2:
            df_head2head = df[
                ((df["Player1"] == player1) & (df["Player2"] == player2))
                | ((df["Player1"] == player2) & (df["Player2"] == player1))
            ]
            if df_head2head.empty:
                st.write(
                    f"No head-to-head matches found between {player1} and {player2}."
                )
            else:
                # Optionally include Elo, Glicko2, TrueSkill for head2head
                # but let's leave it out to keep it simpler.
                generate_analysis_content(df_head2head, include_ratings=False)
        else:
            st.write(
                "Please select two players to compare their head-to-head statistics!",
                color=TERTIARY,
            )


# Run the app
if __name__ == "__main__":
    main()
