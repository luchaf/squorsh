import pandas as pd
import altair as alt
from typing import Tuple
import streamlit as st
from general_utils import meltdown_day_matches
from collections import defaultdict


def chart_matches_over_time(df_in: pd.DataFrame) -> alt.Chart:
    """
    Returns a bar chart showing number of matches over time.
    """
    matches_over_time = df_in.groupby("date").size().reset_index(name="Matches")
    chart = (
        alt.Chart(matches_over_time)
        .mark_bar()
        .encode(x="date:T", y="Matches:Q", tooltip=["date:T", "Matches:Q"])
        .properties(width="container", height=400)
    )
    return chart


def chart_match_distribution(df_in: pd.DataFrame) -> alt.Chart:
    """
    Returns a bar chart showing distribution of final match scores (e.g. 11:0, 11:9, etc.).
    """
    temp_df = df_in.copy()
    temp_df["ResultPair"] = temp_df.apply(
        lambda row: f"{int(max(row['Score1'], row['Score2']))}:{int(min(row['Score1'], row['Score2']))}",
        axis=1,
    )
    pair_counts = temp_df["ResultPair"].value_counts().reset_index()
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
    return results_chart


def get_legendary_matches(df_in: pd.DataFrame, n_closest: int = 10) -> pd.DataFrame:
    """
    Returns the top N 'closest' matches by minimal PointDiff and highest total points.
    """
    temp_df = df_in.copy()
    temp_df["TotalPoints"] = temp_df["Score1"] + temp_df["Score2"]
    # Sort by ascending PointDiff, then descending total points
    df_closest_sorted = temp_df.sort_values(
        ["PointDiff", "TotalPoints"], ascending=[True, False]
    )
    return df_closest_sorted.head(n_closest).copy()


def chart_wins_barchart(df_summary: pd.DataFrame) -> alt.Chart:
    """
    Given a summary DataFrame with 'Player' and 'Wins', returns a bar chart of Wins.
    """
    chart = (
        alt.Chart(df_summary)
        .mark_bar(color="blue")
        .encode(
            x=alt.X("Player:N", sort=list(df_summary["Player"]), title="Player"),
            y=alt.Y("Wins:Q", title="Number of Wins"),
            tooltip=["Player:N", "Wins:Q"],
        )
        .properties(title="Number of Wins by Player", width=700, height=400)
    )
    return chart


def chart_points_barchart(df_summary: pd.DataFrame) -> alt.Chart:
    """
    Given a summary DataFrame with 'Player' and 'Points', returns a bar chart of Points.
    """
    chart = (
        alt.Chart(df_summary)
        .mark_bar(color="orange")
        .encode(
            x=alt.X("Player:N", sort=list(df_summary["Player"]), title="Player"),
            y=alt.Y("Points:Q", title="Total Points"),
            tooltip=["Player:N", "Points:Q"],
        )
        .properties(title="Total Points by Player", width=700, height=400)
    )
    return chart


def chart_wins_over_time(df_in: pd.DataFrame) -> Tuple[alt.Chart, alt.Chart]:
    """
    Returns a tuple (non_cumulative_chart, cumulative_chart) for Wins over time.
    """
    wins_over_time = df_in.groupby(["date", "Winner"]).size().reset_index(name="Wins")
    wins_over_time.rename(columns={"Winner": "Player"}, inplace=True)

    # Non-cumulative
    non_cumulative = (
        alt.Chart(wins_over_time)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Wins:Q", title="Wins Per Day"),
            color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
            tooltip=["date:T", "Player:N", "Wins:Q"],
        )
        .properties(
            title="Non-Cumulative Wins Development Over Time", width=700, height=400
        )
    )

    # Cumulative
    wins_over_time["CumulativeWins"] = wins_over_time.groupby("Player")["Wins"].cumsum()
    cumulative = (
        alt.Chart(wins_over_time)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("CumulativeWins:Q", title="Cumulative Wins"),
            color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
            tooltip=["date:T", "Player:N", "CumulativeWins:Q"],
        )
        .properties(
            title="Cumulative Wins Development Over Time", width=700, height=400
        )
    )

    return non_cumulative, cumulative


def chart_points_over_time(df_in: pd.DataFrame) -> Tuple[alt.Chart, alt.Chart]:
    """
    Returns a tuple (non_cumulative_chart, cumulative_chart) for Points over time.
    """
    points_p1_ot = df_in.groupby(["date", "Player1"])["Score1"].sum().reset_index()
    points_p1_ot.rename(columns={"Player1": "Player", "Score1": "Points"}, inplace=True)

    points_p2_ot = df_in.groupby(["date", "Player2"])["Score2"].sum().reset_index()
    points_p2_ot.rename(columns={"Player2": "Player", "Score2": "Points"}, inplace=True)

    points_over_time = pd.concat([points_p1_ot, points_p2_ot], ignore_index=True)
    points_over_time = (
        points_over_time.groupby(["date", "Player"])["Points"].sum().reset_index()
    )

    # Non-cumulative
    non_cumulative = (
        alt.Chart(points_over_time)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Points:Q", title="Points Per Day"),
            color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
            tooltip=["date:T", "Player:N", "Points:Q"],
        )
        .properties(
            title="Non-Cumulative Points Development Over Time", width=700, height=400
        )
    )

    # Cumulative
    points_over_time["CumulativePoints"] = points_over_time.groupby("Player")[
        "Points"
    ].cumsum()
    cumulative = (
        alt.Chart(points_over_time)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("CumulativePoints:Q", title="Cumulative Points"),
            color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
            tooltip=["date:T", "Player:N", "CumulativePoints:Q"],
        )
        .properties(
            title="Cumulative Points Development Over Time", width=700, height=400
        )
    )

    return non_cumulative, cumulative


def chart_match_intensity_over_time(df_in: pd.DataFrame) -> alt.Chart:
    """
    Returns a line chart of average total points (Score1 + Score2) over time.
    """
    temp = df_in.copy()
    temp["TotalPoints"] = temp["Score1"] + temp["Score2"]
    intensity_over_time = temp.groupby("date")["TotalPoints"].mean().reset_index()
    chart = (
        alt.Chart(intensity_over_time)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("TotalPoints:Q", title="Avg Total Points"),
            tooltip=["date:T", "TotalPoints:Q"],
        )
        .properties(
            title="Average Match Intensity Over Time",
            width=700,
            height=400,
        )
    )
    return chart


def chart_win_rate_by_day_of_week(df_in: pd.DataFrame) -> alt.Chart:
    """
    Returns a heatmap of each player's win rate by day of week.
    """
    # meltdown to get individual rows for each player + day_of_week
    df_melt = meltdown_day_matches(df_in)
    # compute total matches per day-of-week per player
    matches_per_day = (
        df_melt.groupby(["day_of_week", "player"]).size().reset_index(name="matches")
    )
    # compute wins
    wins_per_day = (
        df_melt[df_melt["did_win"] == 1]
        .groupby(["day_of_week", "player"])
        .size()
        .reset_index(name="wins")
    )

    # merge
    merged = pd.merge(
        matches_per_day, wins_per_day, on=["day_of_week", "player"], how="left"
    ).fillna(0)
    merged["win_rate"] = merged["wins"] / merged["matches"]

    # define a custom sort for day_of_week if you want
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    heatmap = (
        alt.Chart(merged)
        .mark_rect()
        .encode(
            x=alt.X("day_of_week:N", sort=day_order, title="Day of Week"),
            y=alt.Y(
                "player:N",
                sort=alt.EncodingSortField(field="win_rate", order="descending"),
                title="Player",
            ),
            color=alt.Color(
                "win_rate:Q", scale=alt.Scale(scheme="greens"), title="Win Rate"
            ),
            tooltip=[
                alt.Tooltip("day_of_week:N", title="Day"),
                alt.Tooltip("player:N", title="Player"),
                alt.Tooltip("win_rate:Q", format=".2f", title="Win Rate"),
                alt.Tooltip("matches:Q", title="Matches"),
                alt.Tooltip("wins:Q", title="Wins"),
            ],
        )
        .properties(title="Win Rate by Day of Week", width=600, height=400)
    )
    return heatmap


def compute_streak_timeseries(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, track a rolling 'streak' value over time.
    +1 for a win, -1 for a loss, cumulatively extended until broken by opposite outcome.
    """
    df_stacked = meltdown_day_matches(df_in).sort_values(["date", "match_number_total"])

    streak_vals = []
    current_streaks = defaultdict(int)
    last_outcome = defaultdict(lambda: None)

    for _, row in df_stacked.iterrows():
        player = row["player"]
        if row["did_win"] == 1:
            if last_outcome[player] == "win":
                current_streaks[player] += 1
            else:
                current_streaks[player] = 1
            last_outcome[player] = "win"
        else:
            if last_outcome[player] == "loss":
                current_streaks[player] -= 1
            else:
                current_streaks[player] = -1
            last_outcome[player] = "loss"

        streak_vals.append(current_streaks[player])

    df_stacked["streak_value"] = streak_vals
    return df_stacked


def chart_streaks_over_time(df_stacked: pd.DataFrame) -> alt.Chart:
    """
    Plot the player's streak_value over time.
    Positive streak_value => consecutive wins.
    Negative => consecutive losses.
    """
    chart = (
        alt.Chart(df_stacked)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("streak_value:Q", title="Streak Value"),
            color="player:N",
            tooltip=["date:T", "player:N", "streak_value:Q"],
        )
        .properties(
            title="Win/Loss Streak Progression Over Time",
            width=800,
            height=400,
        )
    )
    return chart


def display_records_leaderboards(df_in: pd.DataFrame):
    """
    Shows a variety of interesting records/leaderboards:
      - Biggest Blowout (largest PointDiff)
      - Highest Combined Score
      - Most Matches in a Single Day
      - Longest Rivalry (by number of matches)
      - Highest Single-Game Score
    """
    st.subheader("Records & Leaderboards")

    temp = df_in.copy()
    temp["TotalPoints"] = temp["Score1"] + temp["Score2"]

    # 1) Biggest Blowout
    st.markdown("**Biggest Blowout (Largest PointDiff):**")
    biggest_blowout = temp.sort_values("PointDiff", ascending=False).head(1)
    st.dataframe(
        biggest_blowout[
            ["date", "Player1", "Score1", "Player2", "Score2", "PointDiff"]
        ].reset_index(drop=True)
    )

    # 2) Highest Combined Score
    st.markdown("**Highest Combined Score (Longest/Most Intense Match):**")
    highest_score = temp.sort_values("TotalPoints", ascending=False).head(1)
    st.dataframe(
        highest_score[
            ["date", "Player1", "Score1", "Player2", "Score2", "TotalPoints"]
        ].reset_index(drop=True)
    )

    # 3) Most Matches in a Single Day
    st.markdown("**Most Matches in a Single Day:**")
    matches_by_day = temp.groupby("date").size().reset_index(name="Matches")
    busiest_day = matches_by_day.sort_values("Matches", ascending=False).head(1)
    st.dataframe(busiest_day.reset_index(drop=True))

    # 4) Longest Rivalry (by total H2H matches)
    st.markdown("**Longest Rivalry (by total H2H matches):**")
    temp["pair"] = temp.apply(
        lambda row: tuple(sorted([row["Player1"], row["Player2"]])), axis=1
    )
    pair_counts = temp.groupby("pair").size().reset_index(name="match_count")
    top_rivalry = pair_counts.sort_values("match_count", ascending=False).head(1)
    st.dataframe(top_rivalry.reset_index(drop=True))
