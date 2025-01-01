import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
from itertools import combinations
from streamlit_gsheets import GSheetsConnection

# ------------- SETUP -------------
conn = st.connection("gsheets", type=GSheetsConnection)
worksheet_name = "match_results"
df = conn.read(worksheet=worksheet_name)

# ------------- DATA PREPROCESSING -------------
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

# Correct name references
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
df["day_of_week"] = df["date"].dt.day_name()

# ------------- SIDEBAR FILTERS -------------
st.sidebar.header("Filters")

# Date Range Filter (Optional Enhancement)
min_date = df["date"].min()
max_date = df["date"].max()
start_date, end_date = st.sidebar.date_input(
    "Select date range to include",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)
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

# Option to include Elo Ratings
include_elo = st.sidebar.checkbox("Include Elo Ratings", value=True)

# Apply All Filters
df_filtered = df[
    (df["date"] >= start_date)
    & (df["date"] <= end_date)
    & (df["day_of_week"].isin(selected_days))
    & ((df["Player1"].isin(selected_players)) | (df["Player2"].isin(selected_players)))
].copy()


# Function to generate analysis content
def generate_analysis_content(df_input, include_elo):
    tabs = st.tabs(
        [
            "Match Stats",
            "Elo Ratings" if include_elo else None,
            "Wins & Points",
            "Avg. Margin",
            "Win/Loss Streaks",
            "Endurance Metrics",
        ]
    )

    # ------------- 1) MATCH STATS  -------------
    with tabs[0]:
        st.subheader("Overall Match Statistics")
        match_time_tab, match_dist_tab, legendary_tab = st.tabs(
            ["Matches Over Time", "Match Distribution", "Legendary Matches"]
        )

        with match_time_tab:
            matches_over_time = (
                df_input.groupby("date").size().reset_index(name="Matches")
            )
            chart = (
                alt.Chart(matches_over_time)
                .mark_bar()
                .encode(x="date:T", y="Matches:Q", tooltip=["date:T", "Matches:Q"])
                .properties(width="container", height=400)
            )
            st.altair_chart(chart, use_container_width=True)

        with match_dist_tab:
            df_input["ResultPair"] = df_input.apply(
                lambda row: f"{int(max(row['Score1'], row['Score2']))}:{int(min(row['Score1'], row['Score2']))}",
                axis=1,
            )
            pair_counts = df_input["ResultPair"].value_counts().reset_index()
            pair_counts.columns = ["ResultPair", "Count"]

            results_chart = (
                alt.Chart(pair_counts)
                .mark_bar()
                .encode(
                    x=alt.X("Count:Q", title="Number of Matches"),
                    y=alt.Y("ResultPair:N", sort="-x", title="Score Category"),
                    tooltip=["ResultPair", "Count"],
                )
            )
            st.altair_chart(results_chart, use_container_width=True)

        with legendary_tab:
            n_closest = 10
            df_input["TotalPoints"] = df_input["Score1"] + df_input["Score2"]

            df_closest_sorted = df_input.sort_values(
                ["PointDiff", "TotalPoints"], ascending=[True, False]
            )
            closest_subset = df_closest_sorted.head(n_closest)

            closest_subset["date"] = pd.to_datetime(closest_subset["date"]).dt.date

            st.dataframe(
                closest_subset[
                    [
                        "match_number_total",
                        "date",
                        "Player1",
                        "Score1",
                        "Player2",
                        "Score2",
                        "TotalPoints",
                    ]
                ].reset_index(drop=True),
                use_container_width=True,
            )

    # ------------- 2) ELO RATINGS (Optional) -------------
    if include_elo:
        with tabs[1]:
            st.subheader("Elo Ratings")
            df_sorted = df_input.sort_values(["date"], ascending=True)
            elo_ratings = defaultdict(lambda: 1500)
            K = 20

            for _, row in df_sorted.iterrows():
                p1, p2 = row["Player1"], row["Player2"]
                r1, r2 = elo_ratings[p1], elo_ratings[p2]
                exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                exp2 = 1 / (1 + 10 ** ((r1 - r2) / 400))

                if row["Winner"] == p1:
                    elo_ratings[p1] += K * (1 - exp1)
                    elo_ratings[p2] += K * (0 - exp2)
                else:
                    elo_ratings[p1] += K * (0 - exp1)
                    elo_ratings[p2] += K * (1 - exp2)

            elo_df = pd.DataFrame(
                [(player, rating) for player, rating in elo_ratings.items()],
                columns=["Player", "Elo Rating"],
            )
            elo_df.sort_values("Elo Rating", ascending=False, inplace=True)
            st.dataframe(elo_df, use_container_width=True)

    # ------------- Other Tabs (Wins & Points, Avg. Margin, etc.) -------------
    # Add additional subtab processing here following similar patterns.


# ------------- MAIN TABS -------------
main_tab_overall, main_tab_head2head = st.tabs(["Overall Overanalysis", "Head-to-Head"])

# Generate analysis for overall
generate_analysis_content(df_filtered, include_elo)


# Generate analysis for head-to-head
def filter_head_to_head(df, player1, player2):
    return df[
        ((df["Player1"] == player1) & (df["Player2"] == player2))
        | ((df["Player1"] == player2) & (df["Player2"] == player1))
    ]


with main_tab_head2head:
    st.subheader("Head-to-Head Analysis")

    players = sorted(set(df_filtered["Player1"]) | set(df_filtered["Player2"]))
    player1 = st.selectbox("Select Player 1", players, index=0)
    player2 = st.selectbox("Select Player 2", players, index=1)

    df_h2h_filtered = filter_head_to_head(df_filtered, player1, player2)
    generate_analysis_content(df_h2h_filtered, include_elo)
