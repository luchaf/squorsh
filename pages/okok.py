# Refactored Streamlit App
import streamlit as st
import pandas as pd
import altair as alt
from collections import defaultdict
from itertools import combinations
from streamlit_gsheets import GSheetsConnection


# Utility Functions
def preprocess_data(df):
    """Preprocess the data for analysis."""
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df.replace({"Friede": "Friedemann"}, inplace=True)

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
    return df


def apply_filters(df, start_date, end_date, selected_days, selected_players):
    """Apply user-selected filters to the data."""
    return df[
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
        & (df["day_of_week"].isin(selected_days))
        & (
            (df["Player1"].isin(selected_players))
            | (df["Player2"].isin(selected_players))
        )
    ]


def calculate_elo(df):
    """Calculate Elo ratings for players."""
    elo_ratings = defaultdict(lambda: 1500)
    K = 20

    for _, row in df.sort_values("date").iterrows():
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

    return pd.DataFrame(
        [(player, rating) for player, rating in elo_ratings.items()],
        columns=["Player", "Elo Rating"],
    ).sort_values("Elo Rating", ascending=False)


# Streamlit App Setup
st.set_page_config(page_title="Match Analysis App", layout="wide")

# Data Connection
conn = st.connection("gsheets", type=GSheetsConnection)
worksheet_name = "match_results"
df = conn.read(worksheet=worksheet_name)
df = preprocess_data(df)

# Sidebar Filters
st.sidebar.header("Filters")
min_date, max_date = df["date"].min(), df["date"].max()
start_date, end_date = st.sidebar.date_input(
    "Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
)
selected_days = st.sidebar.multiselect(
    "Days of the Week", df["day_of_week"].unique(), default=df["day_of_week"].unique()
)
selected_players = st.sidebar.multiselect(
    "Players",
    sorted(set(df["Player1"]).union(df["Player2"])),
    default=list(sorted(set(df["Player1"]).union(df["Player2"]))),
)

# Apply Filters
df_filtered = apply_filters(
    df,
    pd.to_datetime(start_date),
    pd.to_datetime(end_date),
    selected_days,
    selected_players,
)

# Tabs
main_tab_overall, main_tab_head2head = st.tabs(
    ["Overall Analysis", "Head-to-Head Analysis"]
)


# Overall Analysis
def overall_analysis(df):
    """Generate overall analysis content."""
    with st.container():
        st.subheader("Overall Match Statistics")
        matches_over_time = df.groupby("date").size().reset_index(name="Matches")
        chart = (
            alt.Chart(matches_over_time)
            .mark_bar()
            .encode(x="date:T", y="Matches:Q", tooltip=["date:T", "Matches:Q"])
            .properties(width="container", height=400)
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Elo Ratings")
        elo_df = calculate_elo(df)
        st.dataframe(elo_df, use_container_width=True)


with main_tab_overall:
    overall_analysis(df_filtered)


# Head-to-Head Analysis
def head_to_head_analysis(df, player1, player2):
    """Generate head-to-head analysis content."""
    df_h2h = df[
        ((df["Player1"] == player1) & (df["Player2"] == player2))
        | ((df["Player1"] == player2) & (df["Player2"] == player1))
    ]
    if df_h2h.empty:
        st.write(f"No matches found between {player1} and {player2}.")
    else:
        overall_analysis(df_h2h)


with main_tab_head2head:
    st.subheader("Select Players for Head-to-Head Analysis")
    player1 = st.selectbox(
        "Player 1",
        ["Select..."] + list(sorted(set(df["Player1"].union(df["Player2"])))),
        index=0,
    )
    player2 = st.selectbox(
        "Player 2",
        ["Select..."] + list(sorted(set(df["Player1"].union(df["Player2"])))),
        index=0,
    )

    if player1 != "Select..." and player2 != "Select..." and player1 != player2:
        head_to_head_analysis(df, player1, player2)
    else:
        st.write("Please select two distinct players to compare.")
