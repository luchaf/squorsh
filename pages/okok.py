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

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
df["Player1"] = df["Player1"].replace("Friede", "Friedemann")
df["Player2"] = df["Player2"].replace("Friede", "Friedemann")
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


# ------------- FUNCTIONS -------------
def render_match_stats(df_filtered):
    st.subheader("Overall Match Statistics")
    match_time_tab, match_dist_tab, legendary_tab = st.tabs(
        ["Matches Over Time", "Match Distribution", "Legendary Matches"]
    )

    with match_time_tab:
        matches_over_time = (
            df_filtered.groupby("date").size().reset_index(name="Matches")
        )
        chart = (
            alt.Chart(matches_over_time)
            .mark_bar()
            .encode(x="date:T", y="Matches:Q", tooltip=["date:T", "Matches:Q"])
            .properties(width="container", height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    with match_dist_tab:
        df_filtered["ResultPair"] = df_filtered.apply(
            lambda row: f"{int(max(row['Score1'], row['Score2']))}:{int(min(row['Score1'], row['Score2']))}",
            axis=1,
        )
        pair_counts = df_filtered["ResultPair"].value_counts().reset_index()
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
        df_filtered["TotalPoints"] = df_filtered["Score1"] + df_filtered["Score2"]

        df_closest_sorted = df_filtered.sort_values(
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


def render_elo_ratings(df_filtered):
    st.subheader("Elo Ratings")
    df_sorted = df_filtered.sort_values(["date"], ascending=True)
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


def render_wins_points(df_filtered):
    st.subheader("Wins & Points")
    wins_df = df_filtered.groupby("Winner").size().reset_index(name="Wins")
    points_p1 = df_filtered.groupby("Player1")["Score1"].sum().reset_index()
    points_p1.columns = ["Player", "Points"]
    points_p2 = df_filtered.groupby("Player2")["Score2"].sum().reset_index()
    points_p2.columns = ["Player", "Points"]
    total_points = (
        pd.concat([points_p1, points_p2], ignore_index=True)
        .groupby("Player")["Points"]
        .sum()
        .reset_index()
    )

    summary_df = pd.merge(
        wins_df, total_points, left_on="Winner", right_on="Player", how="outer"
    ).drop(columns="Player")
    summary_df.rename(columns={"Winner": "Player"}, inplace=True)
    summary_df["Wins"] = summary_df["Wins"].fillna(0).astype(int)
    final_summary = pd.merge(
        total_points, summary_df[["Player", "Wins"]], on="Player", how="outer"
    )
    final_summary["Wins"] = final_summary["Wins"].fillna(0).astype(int)
    final_summary.sort_values("Wins", ascending=False, inplace=True, ignore_index=True)

    wins_chart = (
        alt.Chart(final_summary)
        .mark_bar()
        .encode(
            x=alt.X("Player:N", sort=list(final_summary["Player"]), title="Player"),
            y=alt.Y("Wins:Q", title="Number of Wins"),
            tooltip=["Player:N", "Wins:Q"],
        )
        .properties(title="Number of Wins by Player", width=700, height=400)
    )

    points_chart = (
        alt.Chart(final_summary)
        .mark_bar()
        .encode(
            x=alt.X("Player:N", sort=list(final_summary["Player"]), title="Player"),
            y=alt.Y("Points:Q", title="Total Points"),
            tooltip=["Player:N", "Points:Q"],
        )
        .properties(title="Total Points by Player", width=700, height=400)
    )

    st.altair_chart(wins_chart, use_container_width=True)
    st.altair_chart(points_chart, use_container_width=True)


def render_avg_margins(df_filtered):
    st.subheader("Average Margin of Victory & Defeat")

    df_margin_vic = df_filtered.groupby("Winner")["PointDiff"].mean().reset_index()
    df_margin_vic.columns = ["Player", "Avg_margin_victory"]

    df_margin_def = df_filtered.groupby("Loser")["LoserPointDiff"].mean().reset_index()
    df_margin_def.columns = ["Player", "Avg_margin_defeat"]

    df_margin_summary = pd.merge(
        df_margin_vic, df_margin_def, on="Player", how="outer"
    ).fillna(0)

    margin_chart = (
        alt.Chart(df_margin_summary)
        .transform_fold(
            ["Avg_margin_victory", "Avg_margin_defeat"],
            as_=["Metric", "Value"],
        )
        .mark_bar()
        .encode(
            x=alt.X("Player:N", sort="-y", title="Player"),
            y=alt.Y("Value:Q", title="Average Margin"),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["Player:N", "Metric:N", "Value:Q"],
        )
        .properties(
            title="Average Margins for Victory and Defeat",
            height=400,
        )
    )
    st.altair_chart(margin_chart, use_container_width=True)


# ------------- MAIN TABS -------------
main_tab_overall, main_tab_head2head = st.tabs(["Overall Analysis", "Head-to-Head"])

with main_tab_overall:
    subtab_match_stats, subtab_elo, subtab_wins_points, subtab_margins = st.tabs(
        ["Match Stats", "Elo Ratings", "Wins & Points", "Avg. Margin"]
    )

    with subtab_match_stats:
        render_match_stats(df)

    with subtab_elo:
        render_elo_ratings(df)

    with subtab_wins_points:
        render_wins_points(df)

    with subtab_margins:
        render_avg_margins(df)

with main_tab_head2head:
    st.subheader("Select Players for Head-to-Head Analysis")
    players = sorted(set(df["Player1"]) | set(df["Player2"]))

    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Select Player 1", options=players)
    with col2:
        player2 = st.selectbox("Select Player 2", options=players)

    if player1 and player2:
        df_head2head = df[
            ((df["Player1"] == player1) & (df["Player2"] == player2))
            | ((df["Player1"] == player2) & (df["Player2"] == player1))
        ]

        subtab_match_stats, subtab_wins_points, subtab_margins = st.tabs(
            ["Match Stats", "Wins & Points", "Avg. Margin"]
        )

        with subtab_match_stats:
            render_match_stats(df_head2head)

        with subtab_wins_points:
            render_wins_points(df_head2head)

        with subtab_margins:
            render_avg_margins(df_head2head)
