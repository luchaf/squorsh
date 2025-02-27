import pandas as pd
import altair as alt
from typing import Tuple
from dataframe_utils import meltdown_day_matches
from color_palette import (
    BAR_COLOR,
    SECONDARY_BAR_COLOR,
    LINE_OPACITY,
    DESELECTED_OPACITY,
)


def chart_matches_over_time(df_in: pd.DataFrame) -> alt.Chart:
    """
    Returns a bar chart showing number of matches over time.
    """
    matches_over_time = df_in.groupby("date").size().reset_index(name="Matches")
    chart = (
        alt.Chart(matches_over_time)
        .mark_bar(color=BAR_COLOR)
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
        .mark_bar(color=SECONDARY_BAR_COLOR)
        .encode(
            x=alt.X("Count:Q", title="Number of Matches"),
            y=alt.Y("ResultPair:N", sort="-x", title="Score Category"),
            tooltip=["ResultPair", "Count"],
        )
    )
    return results_chart


def chart_wins_barchart(df_summary: pd.DataFrame) -> alt.Chart:
    """
    Given a summary DataFrame with 'Player' and 'Wins', returns a bar chart of Wins.
    """
    chart = (
        alt.Chart(df_summary)
        .mark_bar(color=BAR_COLOR)
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
        .mark_bar(color=SECONDARY_BAR_COLOR)
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

    selection = alt.selection_multi(fields=["Player"], bind="legend")

    # Non-cumulative
    non_cumulative = (
        alt.Chart(wins_over_time)
        .mark_line(opacity=LINE_OPACITY)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Wins:Q", title="Wins Per Day"),
            color=alt.Color(
                "Player:N",
                legend=alt.Legend(title="Player", orient="bottom"),
            ),
            tooltip=["date:T", "Player:N", "Wins:Q"],
            opacity=alt.condition(
                selection, alt.value(LINE_OPACITY), alt.value(DESELECTED_OPACITY)
            ),
        )
        .properties(
            title="Non-Cumulative Wins Development Over Time", width=700, height=400
        )
        .add_selection(selection)
    )

    # Cumulative
    wins_over_time["CumulativeWins"] = wins_over_time.groupby("Player")["Wins"].cumsum()
    cumulative = (
        alt.Chart(wins_over_time)
        .mark_line(opacity=LINE_OPACITY)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("CumulativeWins:Q", title="Cumulative Wins"),
            color=alt.Color(
                "Player:N",
                legend=alt.Legend(title="Player", orient="bottom"),
            ),
            tooltip=["date:T", "Player:N", "CumulativeWins:Q"],
            opacity=alt.condition(
                selection, alt.value(LINE_OPACITY), alt.value(DESELECTED_OPACITY)
            ),
        )
        .properties(
            title="Cumulative Wins Development Over Time", width=700, height=400
        )
        .add_selection(selection)
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

    selection = alt.selection_multi(fields=["Player"], bind="legend")

    # Non-cumulative
    non_cumulative = (
        alt.Chart(points_over_time)
        .mark_line(opacity=LINE_OPACITY)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Points:Q", title="Points Per Day"),
            color=alt.Color(
                "Player:N",
                legend=alt.Legend(title="Player", orient="bottom"),
            ),
            tooltip=["date:T", "Player:N", "Points:Q"],
            opacity=alt.condition(
                selection, alt.value(LINE_OPACITY), alt.value(DESELECTED_OPACITY)
            ),
        )
        .properties(
            title="Non-Cumulative Points Development Over Time", width=700, height=400
        )
        .add_selection(selection)
    )

    # Cumulative
    points_over_time["CumulativePoints"] = points_over_time.groupby("Player")[
        "Points"
    ].cumsum()
    cumulative = (
        alt.Chart(points_over_time)
        .mark_line(opacity=LINE_OPACITY)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("CumulativePoints:Q", title="Cumulative Points"),
            color=alt.Color(
                "Player:N",
                legend=alt.Legend(title="Player", orient="bottom"),
            ),
            tooltip=["date:T", "Player:N", "CumulativePoints:Q"],
            opacity=alt.condition(
                selection, alt.value(LINE_OPACITY), alt.value(DESELECTED_OPACITY)
            ),
        )
        .properties(
            title="Cumulative Points Development Over Time", width=700, height=400
        )
        .add_selection(selection)
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
        .mark_line(point=True, opacity=LINE_OPACITY)
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
            color=alt.Color("win_rate:Q", title="Win Rate"),
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


def chart_streaks_over_time(df_stacked: pd.DataFrame) -> alt.Chart:
    """
    Plot the player's streak_value over time.
    Positive streak_value => consecutive wins.
    Negative => consecutive losses.
    """
    selection = alt.selection_multi(fields=["player"], bind="legend")

    chart = (
        alt.Chart(df_stacked)
        .mark_line(point=True, opacity=LINE_OPACITY)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("streak_value:Q", title="Streak Value"),
            color=alt.Color(
                "player:N",
                legend=alt.Legend(title="Player", orient="bottom"),
            ),
            tooltip=["date:T", "player:N", "streak_value:Q"],
            opacity=alt.condition(
                selection, alt.value(LINE_OPACITY), alt.value(DESELECTED_OPACITY)
            ),
        )
        .properties(
            title="Win/Loss Streak Progression Over Time",
            width=800,
            height=400,
        )
        .add_selection(selection)
    )
    return chart


def chart_win_rate_barchart(df_summary: pd.DataFrame) -> alt.Chart:
    """
    Given a summary DataFrame with 'Player' and 'WinRate', returns a bar chart of Win Rates.
    Handles players with zero wins gracefully.
    """
    # Create a copy to avoid modifying the original
    df = df_summary.copy()

    # Ensure all required columns exist and have proper values
    df["WinRate"] = df["WinRate"].fillna(0)
    df["Wins"] = df["Wins"].fillna(0).astype(int)
    df["Matches"] = df["Matches"].fillna(0).astype(int)

    # Sort by win rate descending
    df = df.sort_values("WinRate", ascending=False)

    chart = (
        alt.Chart(df)
        .mark_bar(color=BAR_COLOR)
        .encode(
            x=alt.X(
                "Player:N",
                sort=list(df["Player"]),  # Use the pre-sorted order
                title="Player",
            ),
            y=alt.Y(
                "WinRate:Q",
                title="Win Rate",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),  # Force scale from 0 to 100%
            ),
            tooltip=[
                alt.Tooltip("Player:N"),
                alt.Tooltip("WinRate:Q", format=".1%"),
                alt.Tooltip("Wins:Q", format="d"),
                alt.Tooltip("Matches:Q", format="d"),
            ],
        )
        .properties(title="Win Rate by Player", width=700, height=400)
    )
    return chart


def chart_win_rate_over_time(df_in: pd.DataFrame) -> Tuple[alt.Chart, alt.Chart]:
    """
    Returns a tuple (non_cumulative_chart, cumulative_chart) for Win Rates over time.
    """
    # Get all unique players
    all_players = sorted(set(df_in["Player1"]) | set(df_in["Player2"]))
    all_dates = pd.date_range(df_in["date"].min(), df_in["date"].max(), freq="D")

    # Create initial daily stats DataFrame with all player-date combinations
    daily_stats = pd.DataFrame(
        [(date, player) for date in all_dates for player in all_players],
        columns=["date", "Player"],
    )

    # Calculate wins for each player per day
    wins = df_in.groupby(["date", "Winner"]).size().reset_index(name="Wins")
    wins.columns = ["date", "Player", "Wins"]

    # Calculate matches for each player per day
    matches_p1 = df_in.groupby(["date", "Player1"]).size().reset_index(name="Matches1")
    matches_p2 = df_in.groupby(["date", "Player2"]).size().reset_index(name="Matches2")
    matches_p1.columns = ["date", "Player", "Matches"]
    matches_p2.columns = ["date", "Player", "Matches"]
    matches = (
        pd.concat([matches_p1, matches_p2])
        .groupby(["date", "Player"])
        .sum()
        .reset_index()
    )

    # Merge everything together
    daily_stats = daily_stats.merge(wins, on=["date", "Player"], how="left").merge(
        matches, on=["date", "Player"], how="left"
    )
    daily_stats.fillna(0, inplace=True)

    # Calculate win rates (handle division by zero)
    daily_stats["WinRate"] = (daily_stats["Wins"] / daily_stats["Matches"]).fillna(0)

    # For the cumulative calculations
    daily_stats["CumWins"] = daily_stats.groupby("Player")["Wins"].cumsum()
    daily_stats["CumMatches"] = daily_stats.groupby("Player")["Matches"].cumsum()
    daily_stats["CumWinRate"] = (
        daily_stats["CumWins"] / daily_stats["CumMatches"]
    ).fillna(0)

    # Remove rows where player had no matches that day for cleaner visualization
    daily_stats = daily_stats[daily_stats["Matches"] > 0]

    selection = alt.selection_multi(fields=["Player"], bind="legend")

    # Non-cumulative chart
    non_cumulative = (
        alt.Chart(daily_stats)
        .mark_line(opacity=LINE_OPACITY)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "WinRate:Q",
                title="Daily Win Rate",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Player:N", legend=alt.Legend(title="Player", orient="bottom")
            ),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("Player:N"),
                alt.Tooltip("WinRate:Q", format=".1%"),
                alt.Tooltip("Wins:Q", format="d"),
                alt.Tooltip("Matches:Q", format="d"),
            ],
            opacity=alt.condition(
                selection, alt.value(LINE_OPACITY), alt.value(DESELECTED_OPACITY)
            ),
        )
        .properties(title="Daily Win Rate Over Time", width=700, height=400)
        .add_selection(selection)
    )

    # Cumulative chart
    cumulative = (
        alt.Chart(daily_stats)
        .mark_line(opacity=LINE_OPACITY)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "CumWinRate:Q",
                title="Cumulative Win Rate",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Player:N", legend=alt.Legend(title="Player", orient="bottom")
            ),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("Player:N"),
                alt.Tooltip("CumWinRate:Q", format=".1%"),
                alt.Tooltip("CumWins:Q", format="d"),
                alt.Tooltip("CumMatches:Q", format="d"),
            ],
            opacity=alt.condition(
                selection, alt.value(LINE_OPACITY), alt.value(DESELECTED_OPACITY)
            ),
        )
        .properties(title="Cumulative Win Rate Over Time", width=700, height=400)
        .add_selection(selection)
    )

    return non_cumulative, cumulative


def chart_win_rate_by_month_of_year(df_in: pd.DataFrame) -> alt.Chart:
    """
    Returns a heatmap of each player's win rate by month of year.
    """
    df_in["month_of_year"] = df_in["date"].dt.strftime("%B")
    df_melt = meltdown_day_matches(df_in)
    df_melt["month_of_year"] = df_melt["date"].dt.strftime("%B")
    matches_per_month = (
        df_melt.groupby(["month_of_year", "player"]).size().reset_index(name="matches")
    )
    wins_per_month = (
        df_melt[df_melt["did_win"] == 1]
        .groupby(["month_of_year", "player"])
        .size()
        .reset_index(name="wins")
    )
    merged = pd.merge(
        matches_per_month, wins_per_month, on=["month_of_year", "player"], how="left"
    ).fillna(0)
    merged["win_rate"] = merged["wins"] / merged["matches"]

    # Ensure all players and months are included
    all_players = sorted(set(df_melt["player"]))
    all_months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    merged = (
        merged.set_index(["month_of_year", "player"])
        .reindex(
            pd.MultiIndex.from_product(
                [all_months, all_players], names=["month_of_year", "player"]
            ),
            fill_value=0,
        )
        .reset_index()
    )

    heatmap = (
        alt.Chart(merged)
        .mark_rect()
        .encode(
            x=alt.X("month_of_year:N", sort=all_months, title="Month of Year"),
            y=alt.Y(
                "player:N",
                sort=alt.EncodingSortField(field="win_rate", order="descending"),
                title="Player",
            ),
            color=alt.Color("win_rate:Q", title="Win Rate"),
            tooltip=[
                alt.Tooltip("month_of_year:N", title="Month"),
                alt.Tooltip("player:N", title="Player"),
                alt.Tooltip("win_rate:Q", format=".2f", title="Win Rate"),
                alt.Tooltip("matches:Q", title="Matches"),
                alt.Tooltip("wins:Q", title="Wins"),
            ],
        )
        .properties(title="Win Rate by Month of Year", width=600, height=400)
    )
    return heatmap


def chart_win_rate_by_year(df_in: pd.DataFrame) -> alt.Chart:
    """
    Returns a heatmap of each player's win rate by year.
    """
    df_in["year"] = df_in["date"].dt.year
    df_melt = meltdown_day_matches(df_in)
    df_melt["year"] = df_melt["date"].dt.year
    matches_per_year = (
        df_melt.groupby(["year", "player"]).size().reset_index(name="matches")
    )
    wins_per_year = (
        df_melt[df_melt["did_win"] == 1]
        .groupby(["year", "player"])
        .size()
        .reset_index(name="wins")
    )
    merged = pd.merge(
        matches_per_year, wins_per_year, on=["year", "player"], how="left"
    ).fillna(0)
    merged["win_rate"] = merged["wins"] / merged["matches"]

    # Ensure all players and years are included
    all_players = sorted(set(df_melt["player"]))
    all_years = sorted(df_in["year"].unique())
    merged = (
        merged.set_index(["year", "player"])
        .reindex(
            pd.MultiIndex.from_product(
                [all_years, all_players], names=["year", "player"]
            ),
            fill_value=0,
        )
        .reset_index()
    )

    heatmap = (
        alt.Chart(merged)
        .mark_rect()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(
                "player:N",
                sort=alt.EncodingSortField(field="win_rate", order="descending"),
                title="Player",
            ),
            color=alt.Color("win_rate:Q", title="Win Rate"),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("player:N", title="Player"),
                alt.Tooltip("win_rate:Q", format=".2f", title="Win Rate"),
                alt.Tooltip("matches:Q", title="Matches"),
                alt.Tooltip("wins:Q", title="Wins"),
            ],
        )
        .properties(title="Win Rate by Year", width=600, height=400)
    )
    return heatmap
