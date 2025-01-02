import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
from itertools import combinations
from streamlit_gsheets import GSheetsConnection
from typing import Tuple, Dict
from functools import lru_cache

# -- NEW: Glicko2 and TrueSkill imports
from glicko2 import Player as Glicko2Player
import trueskill


# ==========================================================
#                  UTILITY & HELPER FUNCTIONS
# ==========================================================


def meltdown_day_matches(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each match into a row per player, preserving original match info.
    Assign each player's 'Nth Match of the Day'.
    Also preserve or recalculate 'day_of_week' for the melted rows to fix KeyError.
    """
    # Ensure 'day_of_week' is present in the original df
    if "day_of_week" not in df_in.columns:
        df_in["day_of_week"] = df_in["date"].dt.day_name()

    df_in = df_in.sort_values(
        ["date", "match_number_total", "match_number_day"], ascending=True
    ).copy()

    # Player1 meltdown
    df_p1 = df_in[
        [
            "date",
            "day_of_week",  # preserve day_of_week
            "Player1",
            "Winner",
            "Loser",
            "Score1",
            "Score2",
            "match_number_total",
            "match_number_day",
        ]
    ].rename(
        columns={
            "Player1": "player",
            "Score1": "score_for_this_player",
            "Score2": "score_for_opponent",
        }
    )
    df_p1["did_win"] = (df_p1["player"] == df_p1["Winner"]).astype(int)

    # Player2 meltdown
    df_p2 = df_in[
        [
            "date",
            "day_of_week",  # preserve day_of_week
            "Player2",
            "Winner",
            "Loser",
            "Score1",
            "Score2",
            "match_number_total",
            "match_number_day",
        ]
    ].rename(
        columns={
            "Player2": "player",
            "Score2": "score_for_this_player",
            "Score1": "score_for_opponent",
        }
    )
    df_p2["did_win"] = (df_p2["player"] == df_p2["Winner"]).astype(int)

    # Combine both
    df_stacked = pd.concat([df_p1, df_p2], ignore_index=True)
    df_stacked = df_stacked.sort_values(
        ["date", "player", "match_number_total", "match_number_day"]
    )
    # Assign Nth match
    df_stacked["MatchOfDay"] = df_stacked.groupby(["date", "player"]).cumcount() + 1

    return df_stacked


def generate_wins_points_summary(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of matches (already filtered) with columns:
    - Winner, Player1, Player2, Score1, Score2
    returns a summary DataFrame with total Wins and total Points.
    """
    # Wins
    wins_df = df_in.groupby("Winner").size().reset_index(name="Wins")

    # Points
    points_p1 = df_in.groupby("Player1")["Score1"].sum().reset_index()
    points_p1.columns = ["Player", "Points"]
    points_p2 = df_in.groupby("Player2")["Score2"].sum().reset_index()
    points_p2.columns = ["Player", "Points"]
    total_points = (
        pd.concat([points_p1, points_p2], ignore_index=True)
        .groupby("Player")["Points"]
        .sum()
        .reset_index()
    )

    # Merge Wins + Points
    summary_df = pd.merge(
        wins_df, total_points, left_on="Winner", right_on="Player", how="outer"
    ).drop(columns="Player")
    summary_df.rename(columns={"Winner": "Player"}, inplace=True)
    summary_df["Wins"] = summary_df["Wins"].fillna(0).astype(int)

    # Create final summary
    final_summary = pd.merge(
        total_points, summary_df[["Player", "Wins"]], on="Player", how="outer"
    )
    final_summary["Wins"] = final_summary["Wins"].fillna(0).astype(int)
    final_summary.dropna(subset=["Player"], inplace=True)
    final_summary.sort_values("Wins", ascending=False, inplace=True, ignore_index=True)

    return final_summary


def generate_elo_ratings(
    df_in: pd.DataFrame, base_elo: float = 1500, K: float = 20
) -> pd.DataFrame:
    """
    Given a DataFrame of matches (already filtered), compute Elo ratings for each player.
    """
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()
    elo_ratings: Dict[str, float] = defaultdict(lambda: base_elo)

    for _, row in df_sorted.iterrows():
        p1, p2 = row["Player1"], row["Player2"]
        r1, r2 = elo_ratings[p1], elo_ratings[p2]
        exp1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        exp2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))

        if row["Winner"] == p1:
            elo_ratings[p1] = r1 + K * (1.0 - exp1)
            elo_ratings[p2] = r2 + K * (0.0 - exp2)
        else:
            elo_ratings[p1] = r1 + K * (0.0 - exp1)
            elo_ratings[p2] = r2 + K * (1.0 - exp2)

    elo_df = pd.DataFrame(
        [(player, rating) for player, rating in elo_ratings.items()],
        columns=["Player", "Elo Rating"],
    )
    elo_df.sort_values("Elo Rating", ascending=False, inplace=True)
    elo_df.reset_index(drop=True, inplace=True)
    return elo_df


# ==========================================================
#            REAL Glicko2 & TrueSkill IMPLEMENTATIONS
# ==========================================================


def generate_glicko2_ratings(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Glicko2 ratings for each player by iterating through matches in chronological order.
    Requires 'pip install glicko2'.
    """
    from glicko2 import Player as Glicko2Player

    # Sort by date and match_number_total to ensure chronological processing
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()

    # Create a dictionary of Glicko2Player objects
    all_players = sorted(set(df_sorted["Player1"]) | set(df_sorted["Player2"]))
    players_dict = {p: Glicko2Player() for p in all_players}

    # Process each match in chronological order
    for _, row in df_sorted.iterrows():
        p1, p2 = row["Player1"], row["Player2"]
        player1 = players_dict[p1]
        player2 = players_dict[p2]

        # Current rating, rating deviation (RD), and volatility
        r1, rd1, vol1 = player1.rating, player1.rd, player1.vol
        r2, rd2, vol2 = player2.rating, player2.rd, player2.vol

        # Determine outcome from perspective of p1
        if row["Winner"] == p1:
            score_p1, score_p2 = 1.0, 0.0
        else:
            score_p1, score_p2 = 0.0, 1.0

        # Update p1's rating with p2 as the opponent
        player1.update_player([r2], [rd2], [score_p1])
        # Update p2's rating with p1 as the opponent
        player2.update_player([r1], [rd1], [score_p2])

    # Build output DataFrame
    results = []
    for p in all_players:
        pl = players_dict[p]
        results.append(
            {
                "Player": p,
                "Glicko2 Rating": pl.rating,
                "RD": pl.rd,
                "Volatility": pl.vol,
            }
        )
    df_glicko = pd.DataFrame(results)
    df_glicko.sort_values("Glicko2 Rating", ascending=False, inplace=True)
    df_glicko.reset_index(drop=True, inplace=True)
    return df_glicko


def generate_trueskill_ratings(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TrueSkill ratings by iterating matches in chronological order.
    By default, each player starts with Rating(mu=25, sigma=8.333...).
    """
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()

    # Dictionary of player -> trueskill.Rating()
    players_dict = {}
    all_players = sorted(set(df_sorted["Player1"]) | set(df_sorted["Player2"]))
    for p in all_players:
        players_dict[p] = trueskill.Rating()  # default mu=25, sigma=8.333

    for _, row in df_sorted.iterrows():
        p1, p2 = row["Player1"], row["Player2"]
        rating_p1 = players_dict[p1]
        rating_p2 = players_dict[p2]

        # If p1 is the winner, we do
        if row["Winner"] == p1:
            rating_p1, rating_p2 = trueskill.rate_1vs1(rating_p1, rating_p2)
        else:
            # p2 is the winner, so invert the order
            rating_p2, rating_p1 = trueskill.rate_1vs1(rating_p2, rating_p1)

        players_dict[p1] = rating_p1
        players_dict[p2] = rating_p2

    data_out = []
    for p in all_players:
        r = players_dict[p]
        data_out.append(
            {
                "Player": p,
                "TrueSkill Mu": r.mu,
                "TrueSkill Sigma": r.sigma,
                "TrueSkill Rating": r.mu - 3 * r.sigma,  # conservative rating
            }
        )
    df_ts = pd.DataFrame(data_out)
    # Sort by the conservative skill estimate
    df_ts.sort_values("TrueSkill Rating", ascending=False, inplace=True)
    df_ts.reset_index(drop=True, inplace=True)
    return df_ts


# ==========================================================
#          CORE VISUALIZATION COMPONENTS (CHARTS/TABLES)
# ==========================================================


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


# ==========================================================
#                  NEW FEATURES
# ==========================================================

# 1) MATCH PACE / INTENSITY


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


# 3) PERFORMANCE BY DAY OF WEEK (DETAILED)


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


# 4) HOT & COLD STREAKS OVER TIME


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


# 11) RECORDS / LEADERBOARDS


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

    # 5) Highest Single-Game Score
    st.markdown("**Highest Single-Game Score:**")
    temp["max_score"] = temp[["Score1", "Score2"]].max(axis=1)
    highest_single_game = temp.sort_values("max_score", ascending=False).head(1)
    st.dataframe(
        highest_single_game[
            ["date", "Player1", "Score1", "Player2", "Score2", "max_score"]
        ].reset_index(drop=True)
    )


# ==========================================================
#                  ANALYSIS SUB-SECTIONS
# ==========================================================


def display_match_stats(df_filtered: pd.DataFrame):
    """
    Section: "Match Stats"
    Subtabs:
      1) "Matches Over Time"
      2) "Match Distribution"
      3) "Legendary Matches"
      4) "Match Intensity"
      5) "Day-of-Week Performance"
    """
    (
        match_time_tab,
        match_dist_tab,
        legendary_tab,
        intensity_tab,
        dayofweek_tab,
    ) = st.tabs(
        [
            "Matches Over Time",
            "Match Distribution",
            "Legendary Matches",
            "Match Intensity",
            "Day-of-Week Performance",
        ]
    )

    with match_time_tab:
        st.subheader("Matches Over Time")
        st.altair_chart(chart_matches_over_time(df_filtered), use_container_width=True)

    with match_dist_tab:
        st.subheader("Match Result Distribution")
        st.altair_chart(chart_match_distribution(df_filtered), use_container_width=True)

    with legendary_tab:
        st.subheader("The Ten Most Legendary Matches")
        legendary_df = get_legendary_matches(df_filtered, n_closest=10)
        legendary_df["date"] = pd.to_datetime(legendary_df["date"]).dt.date
        st.dataframe(
            legendary_df[
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

    with intensity_tab:
        st.subheader("Match Intensity (Average Total Points Over Time)")
        st.altair_chart(
            chart_match_intensity_over_time(df_filtered), use_container_width=True
        )

        # Optional: Top 10 Most Intense Matches
        temp = df_filtered.copy()
        temp["TotalPoints"] = temp["Score1"] + temp["Score2"]
        temp.sort_values("TotalPoints", ascending=False, inplace=True)
        st.markdown("**Top 10 Most Intense Matches by Total Points**")
        st.dataframe(
            temp.head(10)[
                ["date", "Player1", "Score1", "Player2", "Score2", "TotalPoints"]
            ].reset_index(drop=True)
        )

    with dayofweek_tab:
        st.subheader("Detailed Win Rate by Day of Week")
        st.altair_chart(
            chart_win_rate_by_day_of_week(df_filtered), use_container_width=True
        )
        st.write(
            "Hover over the chart to see each player's matches, wins, and computed win rate by day of the week."
        )


def display_elo_and_alternative_ratings(df_filtered: pd.DataFrame):
    """
    Section: "Ratings Systems"
    Subtabs:
      1) "Elo Ratings"
      2) "Glicko2 Ratings"
      3) "TrueSkill Ratings"
    """
    elo_tab, glicko_tab, trueskill_tab = st.tabs(["Elo", "Glicko2", "TrueSkill"])

    with elo_tab:
        st.subheader("Elo Ratings")
        elo_df = generate_elo_ratings(df_filtered, base_elo=1500, K=20)
        st.dataframe(elo_df, use_container_width=True)

    with glicko_tab:
        st.subheader("Glicko2 Ratings")
        glicko_df = generate_glicko2_ratings(df_filtered)
        st.dataframe(glicko_df, use_container_width=True)
        st.markdown(
            "Above are the **live** Glicko2 ratings after processing each match in chronological order."
        )

    with trueskill_tab:
        st.subheader("TrueSkill Ratings")
        ts_df = generate_trueskill_ratings(df_filtered)
        st.dataframe(ts_df, use_container_width=True)
        st.markdown(
            "Column **TrueSkill Rating** = `mu - 3*sigma`, a conservative estimate."
        )


def display_wins_and_points(df_filtered: pd.DataFrame):
    """
    Section: "Wins & Points"
    Subtabs: "Wins" & "Points"
    Each has "Current Standings" and "Trends Over Time"
    """
    st.subheader("Wins & Points")

    # Summaries
    final_summary = generate_wins_points_summary(df_filtered)
    final_summary_wins = final_summary.copy()
    final_summary_points = final_summary.copy()
    final_summary_wins.sort_values(by="Wins", ascending=False, inplace=True)
    final_summary_points.sort_values(by="Points", ascending=False, inplace=True)

    # Charts
    wins_chart = chart_wins_barchart(final_summary_wins)
    points_chart = chart_points_barchart(final_summary_points)
    non_cum_wins, cum_wins = chart_wins_over_time(df_filtered)
    non_cum_points, cum_points = chart_points_over_time(df_filtered)

    chart_tab_wins, chart_tab_points = st.tabs(["Wins", "Points"])

    # --- Wins Tab ---
    with chart_tab_wins:
        subtab_curr, subtab_trend = st.tabs(["Current Standings", "Trends Over Time"])
        with subtab_curr:
            st.subheader("Wins per Player (Current)")
            st.altair_chart(wins_chart, use_container_width=True)
        with subtab_trend:
            subtab_non_cum, subtab_cum = st.tabs(["Non-Cumulative", "Cumulative"])
            with subtab_non_cum:
                st.subheader("Non-Cumulative Wins Over Time")
                st.altair_chart(non_cum_wins, use_container_width=True)
            with subtab_cum:
                st.subheader("Cumulative Wins Over Time")
                st.altair_chart(cum_wins, use_container_width=True)

    # --- Points Tab ---
    with chart_tab_points:
        subtab_curr, subtab_trend = st.tabs(["Current Standings", "Trends Over Time"])
        with subtab_curr:
            st.subheader("Points per Player (Current)")
            st.altair_chart(points_chart, use_container_width=True)
        with subtab_trend:
            subtab_non_cum, subtab_cum = st.tabs(["Non-Cumulative", "Cumulative"])
            with subtab_non_cum:
                st.subheader("Non-Cumulative Points Over Time")
                st.altair_chart(non_cum_points, use_container_width=True)
            with subtab_cum:
                st.subheader("Cumulative Points Over Time")
                st.altair_chart(cum_points, use_container_width=True)


def display_avg_margin(df_filtered: pd.DataFrame):
    """
    Section: "Avg. Margin"
    Subtabs: "Current Standings", "Trends Over Time"
    Each with "Avg Margin of Victory" and "Avg Margin of Defeat"
    """
    st.subheader("Average Margin of Victory & Defeat")

    # Prepare data
    df_margin_vic = df_filtered.groupby("Winner")["PointDiff"].mean().reset_index()
    df_margin_vic.columns = ["Player", "Avg_margin_victory"]

    df_margin_def = df_filtered.groupby("Loser")["LoserPointDiff"].mean().reset_index()
    df_margin_def.columns = ["Player", "Avg_margin_defeat"]

    df_margin_summary = pd.merge(
        df_margin_vic, df_margin_def, on="Player", how="outer"
    ).fillna(0)

    margin_tabs = st.tabs(["Current Standings", "Trends Over Time"])

    # --- Current Standings ---
    with margin_tabs[0]:
        st.subheader("Current Standings: Average Margins")
        margin_chart = (
            alt.Chart(df_margin_summary)
            .transform_fold(
                ["Avg_margin_victory", "Avg_margin_defeat"], as_=["Metric", "Value"]
            )
            .mark_bar()
            .encode(
                x=alt.X("Player:N", sort="-y", title="Player"),
                y=alt.Y("Value:Q", title="Average Margin"),
                color=alt.Color("Metric:N", title="Metric"),
                tooltip=["Player:N", "Metric:N", "Value:Q"],
            )
            .properties(title="Average Margins for Victory and Defeat", height=400)
        )
        st.altair_chart(margin_chart, use_container_width=True)

    # --- Trends Over Time ---
    with margin_tabs[1]:
        st.subheader("Trends Over Time: Average Margins")
        avg_margin_victory_tab, avg_margin_defeat_tab = st.tabs(
            ["Avg. Margin of Victory", "Avg. Margin of Defeat"]
        )

        with avg_margin_victory_tab:
            df_margin_vic2 = (
                df_filtered.groupby(["date", "Winner"])["PointDiff"]
                .mean()
                .reset_index()
                .rename(columns={"Winner": "Player", "PointDiff": "Avg_margin_victory"})
            )
            victory_chart = (
                alt.Chart(df_margin_vic2)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Avg_margin_victory:Q", title="Average Margin of Victory"),
                    color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                    tooltip=["date:T", "Player:N", "Avg_margin_victory:Q"],
                )
                .properties(
                    title="Trends in Average Margin of Victory Over Time",
                    width=700,
                    height=400,
                )
            )
            st.altair_chart(victory_chart, use_container_width=True)

        with avg_margin_defeat_tab:
            df_margin_def2 = (
                df_filtered.groupby(["date", "Loser"])["LoserPointDiff"]
                .mean()
                .reset_index()
                .rename(
                    columns={"Loser": "Player", "LoserPointDiff": "Avg_margin_defeat"}
                )
            )
            defeat_chart = (
                alt.Chart(df_margin_def2)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Avg_margin_defeat:Q", title="Average Margin of Defeat"),
                    color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                    tooltip=["date:T", "Player:N", "Avg_margin_defeat:Q"],
                )
                .properties(
                    title="Trends in Average Margin of Defeat Over Time",
                    width=700,
                    height=400,
                )
            )
            st.altair_chart(defeat_chart, use_container_width=True)


def display_win_loss_streaks(df_filtered: pd.DataFrame):
    """
    Section: "Win/Loss Streaks"
    Subtabs:
      1) "Longest Streaks (Overall)"
      2) "Streaks Over Time"
    """
    st.subheader("Winning and Losing Streaks")

    tabs_streaks = st.tabs(
        ["Longest Streaks (Overall)", "Hot & Cold Streaks Over Time"]
    )

    # ---- Tab 1: Overall Longest Streaks ----
    with tabs_streaks[0]:
        df_sorted = df_filtered.sort_values(
            ["date", "match_number_total"], ascending=True
        )
        streaks = []
        unique_players = sorted(
            set(df_filtered["Player1"]) | set(df_filtered["Player2"])
        )

        for player in unique_players:
            current_win, max_win = 0, 0
            current_loss, max_loss = 0, 0

            for _, row in df_sorted.iterrows():
                if row["Winner"] == player:
                    current_win += 1
                    max_win = max(max_win, current_win)
                    current_loss = 0
                elif row["Loser"] == player:
                    current_loss += 1
                    max_loss = max(max_loss, current_loss)
                    current_win = 0

            streaks.append((player, max_win, max_loss))

        streaks_df = pd.DataFrame(
            streaks, columns=["Player", "Longest Win Streak", "Longest Loss Streak"]
        )
        streaks_df.sort_values("Longest Win Streak", ascending=False, inplace=True)
        st.dataframe(streaks_df, use_container_width=True)

    # ---- Tab 2: Streaks Over Time Visualization ----
    with tabs_streaks[1]:
        df_stacked_streaks = compute_streak_timeseries(df_filtered)
        st.altair_chart(
            chart_streaks_over_time(df_stacked_streaks), use_container_width=True
        )
        st.markdown(
            """
            **Interpretation**:
            - Positive streak_value indicates consecutive wins.
            - Negative streak_value indicates consecutive losses.
            """
        )


def display_endurance_and_grit(df_filtered: pd.DataFrame):
    """
    Section: "Endurance and Grit"
    Subtabs:
      1) "N-th Match of the Day"
      2) "Balls of Steel" (filter: 11:9 or 9:11)
      3) "Balls of Adamantium" (filter: >= 12:10 or >= 10:12)
    """
    endurance_tabs = st.tabs(
        ["N-th Match of the Day", "Balls of Steel", "Balls of Adamantium"]
    )
    df_backup = df_filtered.copy()

    # --- 1) N-th Match of the Day ---
    with endurance_tabs[0]:
        st.subheader("Endurance Metrics: Performance by N-th Match of Day")
        df_daycount = meltdown_day_matches(df_filtered)

        df_day_agg = (
            df_daycount.groupby(["player", "MatchOfDay"])["did_win"]
            .agg(["sum", "count"])
            .reset_index()
        )
        df_day_agg["win_rate"] = df_day_agg["sum"] / df_day_agg["count"]

        base = alt.Chart(df_day_agg).encode(
            x=alt.X("MatchOfDay:Q", title="Nth Match of the Day"),
            y=alt.Y("win_rate:Q", title="Win Rate (0-1)"),
            color=alt.Color("player:N", title="Player"),
            tooltip=[
                alt.Tooltip("player:N"),
                alt.Tooltip("MatchOfDay:Q"),
                alt.Tooltip("win_rate:Q", format=".2f"),
                alt.Tooltip("sum:Q", title="Wins"),
                alt.Tooltip("count:Q", title="Matches"),
            ],
        )

        lines_layer = base.mark_line(point=True)
        trend_layer = (
            base.transform_regression("MatchOfDay", "win_rate", groupby=["player"])
            .mark_line(strokeDash=[4, 4])
            .encode(opacity=alt.value(0.7))
        )
        chart_match_of_day = alt.layer(lines_layer, trend_layer).properties(
            width="container", height=400
        )
        st.altair_chart(chart_match_of_day, use_container_width=True)

        st.markdown(
            """
            This chart shows how each **selected** player performs in their 1st, 2nd, 3rd, etc. match **per day**.  
            The **solid line** is their actual data, and the **dashed line** is a linear trend line.
            """
        )

    # --- 2) Balls of Steel: 11:9 or 9:11 ---
    with endurance_tabs[1]:
        df_steel = df_backup[
            (
                ((df_backup["Score1"] == 11) & (df_backup["Score2"] == 9))
                | ((df_backup["Score1"] == 9) & (df_backup["Score2"] == 11))
            )
        ].copy()

        if df_steel.empty:
            st.warning(
                "No matches ended with a tight 11:9 or 9:11 score under current filters."
            )
        else:
            # Summaries
            final_summary_steel = generate_wins_points_summary(df_steel)
            final_summary_wins_steel = final_summary_steel.copy()
            final_summary_points_steel = final_summary_steel.copy()
            final_summary_wins_steel.sort_values(
                by="Wins", ascending=False, inplace=True
            )
            final_summary_points_steel.sort_values(
                by="Points", ascending=False, inplace=True
            )

            # Charts
            wins_chart_steel = chart_wins_barchart(final_summary_wins_steel)
            points_chart_steel = chart_points_barchart(final_summary_points_steel)
            non_cum_wins_steel, cum_wins_steel = chart_wins_over_time(df_steel)
            non_cum_points_steel, cum_points_steel = chart_points_over_time(df_steel)

            chart_tab_wins_steel, chart_tab_points_steel = st.tabs(["Wins", "Points"])

            # Wins Tab
            with chart_tab_wins_steel:
                subtab_curr, subtab_trend = st.tabs(
                    ["Current Standings", "Trends Over Time"]
                )
                with subtab_curr:
                    st.subheader("Wins per Player (Current)")
                    st.altair_chart(wins_chart_steel, use_container_width=True)
                with subtab_trend:
                    subtab_non_cum, subtab_cum = st.tabs(
                        ["Non-Cumulative", "Cumulative"]
                    )
                    with subtab_non_cum:
                        st.subheader("Non-Cumulative Wins")
                        st.altair_chart(non_cum_wins_steel, use_container_width=True)
                    with subtab_cum:
                        st.subheader("Cumulative Wins")
                        st.altair_chart(cum_wins_steel, use_container_width=True)

            # Points Tab
            with chart_tab_points_steel:
                subtab_curr, subtab_trend = st.tabs(
                    ["Current Standings", "Trends Over Time"]
                )
                with subtab_curr:
                    st.subheader("Points per Player (Current)")
                    st.altair_chart(points_chart_steel, use_container_width=True)
                with subtab_trend:
                    subtab_non_cum, subtab_cum = st.tabs(
                        ["Non-Cumulative", "Cumulative"]
                    )
                    with subtab_non_cum:
                        st.subheader("Non-Cumulative Points")
                        st.altair_chart(non_cum_points_steel, use_container_width=True)
                    with subtab_cum:
                        st.subheader("Cumulative Points")
                        st.altair_chart(cum_points_steel, use_container_width=True)

    # --- 3) Balls of Adamantium: >=12:10 or >=10:12 ---
    with endurance_tabs[2]:
        df_adamantium = df_backup[
            (
                ((df_backup["Score1"] >= 12) & (df_backup["Score2"] >= 10))
                | ((df_backup["Score1"] >= 10) & (df_backup["Score2"] >= 12))
            )
        ].copy()

        if df_adamantium.empty:
            st.warning(
                "No matches ended with >=12:10 or >=10:12 under current filters."
            )
        else:
            # Summaries
            final_summary_adam = generate_wins_points_summary(df_adamantium)
            final_summary_wins_adam = final_summary_adam.copy()
            final_summary_points_adam = final_summary_adam.copy()
            final_summary_wins_adam.sort_values(
                by="Wins", ascending=False, inplace=True
            )
            final_summary_points_adam.sort_values(
                by="Points", ascending=False, inplace=True
            )

            # Charts
            wins_chart_adam = chart_wins_barchart(final_summary_wins_adam)
            points_chart_adam = chart_points_barchart(final_summary_points_adam)
            non_cum_wins_adam, cum_wins_adam = chart_wins_over_time(df_adamantium)
            non_cum_points_adam, cum_points_adam = chart_points_over_time(df_adamantium)

            chart_tab_wins_adam, chart_tab_points_adam = st.tabs(["Wins", "Points"])

            # Wins Tab
            with chart_tab_wins_adam:
                subtab_curr, subtab_trend = st.tabs(
                    ["Current Standings", "Trends Over Time"]
                )
                with subtab_curr:
                    st.subheader("Wins per Player (Current)")
                    st.altair_chart(wins_chart_adam, use_container_width=True)
                with subtab_trend:
                    subtab_non_cum, subtab_cum = st.tabs(
                        ["Non-Cumulative", "Cumulative"]
                    )
                    with subtab_non_cum:
                        st.subheader("Non-Cumulative Wins")
                        st.altair_chart(non_cum_wins_adam, use_container_width=True)
                    with subtab_cum:
                        st.subheader("Cumulative Wins")
                        st.altair_chart(cum_wins_adam, use_container_width=True)

            # Points Tab
            with chart_tab_points_adam:
                subtab_curr, subtab_trend = st.tabs(
                    ["Current Standings", "Trends Over Time"]
                )
                with subtab_curr:
                    st.subheader("Points per Player (Current)")
                    st.altair_chart(points_chart_adam, use_container_width=True)
                with subtab_trend:
                    subtab_non_cum, subtab_cum = st.tabs(
                        ["Non-Cumulative", "Cumulative"]
                    )
                    with subtab_non_cum:
                        st.subheader("Non-Cumulative Points")
                        st.altair_chart(non_cum_points_adam, use_container_width=True)
                    with subtab_cum:
                        st.subheader("Cumulative Points")
                        st.altair_chart(cum_points_adam, use_container_width=True)


def generate_analysis_content(df_filtered: pd.DataFrame, include_elo: bool):
    """
    Creates the sub-tabs for Overall Overanalysis:
      1) Match Stats
      2) Ratings (Elo, Glicko2, TrueSkill) (if include_elo=True)
      3) Wins & Points
      4) Avg. Margin
      5) Win/Loss Streaks
      6) Endurance and Grit
      7) Records & Leaderboards
    """
    list_of_tabs = [
        "Match Stats",
        "Ratings",
        "Wins & Points",
        "Avg. Margin",
        "Win/Loss Streaks",
        "Endurance and Grit",
        "Records & Leaderboards",
    ]
    if not include_elo:
        # If we skip the rating systems entirely
        list_of_tabs.remove("Ratings")

    tabs = st.tabs(list_of_tabs)
    idx = 0

    # 1) MATCH STATS
    with tabs[idx]:
        display_match_stats(df_filtered)
    idx += 1

    # 2) RATINGS (Optional)
    if include_elo:
        with tabs[idx]:
            display_elo_and_alternative_ratings(df_filtered)
        idx += 1

    # 3) WINS & POINTS
    with tabs[idx]:
        display_wins_and_points(df_filtered)
    idx += 1

    # 4) AVERAGE MARGIN
    with tabs[idx]:
        display_avg_margin(df_filtered)
    idx += 1

    # 5) WIN/LOSS STREAKS
    with tabs[idx]:
        display_win_loss_streaks(df_filtered)
    idx += 1

    # 6) ENDURANCE & GRIT
    with tabs[idx]:
        display_endurance_and_grit(df_filtered)
    idx += 1

    # 7) RECORDS & LEADERBOARDS
    with tabs[idx]:
        display_records_leaderboards(df_filtered)
    idx += 1


# ==========================================================
#                     MAIN APP
# ==========================================================
def main():
    # ------------- SETUP -------------
    st.set_page_config(layout="wide")
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
    main_tab_overall, main_tab_head2head = st.tabs(
        ["Overall Overanalysis", "Head-to-Head"]
    )

    # Overall Analysis Tab
    with main_tab_overall:
        generate_analysis_content(df_filtered, include_elo=True)

    # Head-to-Head Analysis Tab
    with main_tab_head2head:
        st.subheader("Select Players for Head-to-Head Analysis")
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
                generate_analysis_content(df_head2head, include_elo=False)
        else:
            st.write(
                "Please select two players to compare their head-to-head statistics!"
            )


# Run the app
if __name__ == "__main__":
    main()
