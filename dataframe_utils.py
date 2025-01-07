import pandas as pd
from collections import defaultdict


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
    returns a summary DataFrame with total Wins, total Points, and total Matches.
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

    # Matches
    matches_p1 = df_in.groupby("Player1").size().reset_index(name="Matches")
    matches_p1.columns = ["Player", "Matches"]
    matches_p2 = df_in.groupby("Player2").size().reset_index(name="Matches")
    matches_p2.columns = ["Player", "Matches"]
    total_matches = (
        pd.concat([matches_p1, matches_p2], ignore_index=True)
        .groupby("Player")["Matches"]
        .sum()
        .reset_index()
    )

    # Merge Wins, Points, and Matches
    summary_df = pd.merge(
        wins_df, total_points, left_on="Winner", right_on="Player", how="outer"
    ).drop(columns="Player")
    summary_df.rename(columns={"Winner": "Player"}, inplace=True)
    summary_df["Wins"] = summary_df["Wins"].fillna(0).astype(int)

    final_summary = pd.merge(
        summary_df, total_matches, on="Player", how="outer"
    ).fillna(0)
    final_summary["Wins"] = final_summary["Wins"].astype(int)
    final_summary["Matches"] = final_summary["Matches"].astype(int)
    final_summary.dropna(subset=["Player"], inplace=True)
    final_summary.sort_values("Wins", ascending=False, inplace=True, ignore_index=True)

    return final_summary


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
