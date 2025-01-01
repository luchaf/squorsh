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

# Apply All Filters
df_filtered = df[
    (df["date"] >= start_date)
    & (df["date"] <= end_date)
    & (df["day_of_week"].isin(selected_days))
    & ((df["Player1"].isin(selected_players)) | (df["Player2"].isin(selected_players)))
].copy()

# ------------- MAIN TABS -------------
main_tab_overall, main_tab_head2head = st.tabs(["Overall Analysis", "Head-to-Head"])

# ==========================================================
#                    OVERALL ANALYSIS
# ==========================================================
with main_tab_overall:
    (
        subtab_match_stats,  # Consolidated tab for Matches Over Time, Distribution, Legendary
        subtab_elo,
        subtab_wins_points,
        subtab_margins,
        subtab_streaks,
        subtab_endurance,
    ) = st.tabs(
        [
            "Match Stats",  # 1) Combined tab for match-related data
            "Elo Ratings",  # 2) Elo
            "Wins & Points",  # 3) Wins & Points
            "Avg. Margin",  # 4) Average Margin
            "Win/Loss Streaks",  # 5) Streaks
            "Endurance Metrics",  # 6) Endurance (Nth match of the day)
        ]
    )

    # ------------- 1) MATCH STATS  -------------
    with subtab_match_stats:
        st.subheader("Overall Match Statistics")

        # Three subtabs for Over Time, Distribution, Legendary
        match_time_tab, match_dist_tab, legendary_tab = st.tabs(
            ["Matches Over Time", "Match Distribution", "Legendary Matches"]
        )

        # ---- 1a) Matches Over Time ----
        with match_time_tab:
            st.subheader("Matches Over Time")
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

        # ---- 1b) Match Result Distribution ----
        with match_dist_tab:
            st.subheader("Match Result Distribution")
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

        # ---- 1c) Legendary Matches ----
        with legendary_tab:
            st.subheader("The Ten Most Legendary Matches")
            n_closest = 10
            df_filtered["TotalPoints"] = df_filtered["Score1"] + df_filtered["Score2"]

            # Sort by margin ascending, then total points descending
            df_closest_sorted = df_filtered.sort_values(
                ["PointDiff", "TotalPoints"], ascending=[True, False]
            )
            closest_subset = df_closest_sorted.head(n_closest)

            # Ensure date is in date format only
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

    # ------------- 2) ELO RATINGS  -------------
    with subtab_elo:
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
            columns=["Player", "Elo_Rating"],
        )
        elo_df.sort_values("Elo_Rating", ascending=False, inplace=True)
        st.dataframe(elo_df, use_container_width=True)

    # ------------- 3) WINS & POINTS  -------------
    with subtab_wins_points:
        st.subheader("Wins & Points")

        # Wins & Points Summary
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
        final_summary.sort_values(
            "Wins", ascending=False, inplace=True, ignore_index=True
        )

        final_summary_wins = final_summary.copy()
        final_summary_points = final_summary.copy()
        final_summary_wins.sort_values(by="Wins", ascending=False, inplace=True)
        final_summary_points.sort_values(by="Points", ascending=False, inplace=True)

        # Charts: Wins & Points (Current Standings)
        wins_chart = (
            alt.Chart(final_summary_wins)
            .mark_bar(color="blue")
            .encode(
                x=alt.X(
                    "Player:N", sort=list(final_summary_wins["Player"]), title="Player"
                ),
                y=alt.Y("Wins:Q", title="Number of Wins"),
                tooltip=["Player:N", "Wins:Q"],
            )
            .properties(title="Number of Wins by Player", width=700, height=400)
        )

        points_chart = (
            alt.Chart(final_summary_points)
            .mark_bar(color="orange")
            .encode(
                x=alt.X(
                    "Player:N",
                    sort=list(final_summary_points["Player"]),
                    title="Player",
                ),
                y=alt.Y("Points:Q", title="Total Points"),
                tooltip=["Player:N", "Points:Q"],
            )
            .properties(title="Total Points by Player", width=700, height=400)
        )

        # Over Time (Wins & Points)
        wins_over_time = (
            df_filtered.groupby(["date", "Winner"]).size().reset_index(name="Wins")
        )
        wins_over_time.rename(columns={"Winner": "Player"}, inplace=True)

        points_p1_ot = (
            df_filtered.groupby(["date", "Player1"])["Score1"].sum().reset_index()
        )
        points_p2_ot = (
            df_filtered.groupby(["date", "Player2"])["Score2"].sum().reset_index()
        )

        points_p1_ot.rename(
            columns={"Player1": "Player", "Score1": "Points"}, inplace=True
        )
        points_p2_ot.rename(
            columns={"Player2": "Player", "Score2": "Points"}, inplace=True
        )
        points_over_time = pd.concat([points_p1_ot, points_p2_ot], ignore_index=True)
        points_over_time = (
            points_over_time.groupby(["date", "Player"])["Points"].sum().reset_index()
        )

        # Non-cumulative vs. cumulative
        wins_over_time["CumulativeWins"] = wins_over_time.groupby("Player")[
            "Wins"
        ].cumsum()
        points_over_time["CumulativePoints"] = points_over_time.groupby("Player")[
            "Points"
        ].cumsum()

        # Non-cumulative charts
        non_cumulative_wins_chart = (
            alt.Chart(wins_over_time)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Wins:Q", title="Wins Per Match"),
                color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                tooltip=["date:T", "Player:N", "Wins:Q"],
            )
            .properties(
                title="Non-Cumulative Wins Development Over Time", width=700, height=400
            )
        )

        non_cumulative_points_chart = (
            alt.Chart(points_over_time)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Points:Q", title="Points Per Match"),
                color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                tooltip=["date:T", "Player:N", "Points:Q"],
            )
            .properties(
                title="Non-Cumulative Points Development Over Time",
                width=700,
                height=400,
            )
        )

        # Cumulative charts
        cumulative_wins_chart = (
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

        cumulative_points_chart = (
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

        # Display in sub-tabs
        chart_tab_wins, chart_tab_points = st.tabs(["Wins", "Points"])

        # --- Wins Tab ---
        with chart_tab_wins:
            subtab_curr, subtab_trend = st.tabs(
                ["Current Standings", "Trends Over Time"]
            )
            with subtab_curr:
                st.subheader("Wins per Player (Current)")
                st.altair_chart(wins_chart, use_container_width=True)
            with subtab_trend:
                subtab_non_cum, subtab_cum = st.tabs(["Non-Cumulative", "Cumulative"])
                with subtab_non_cum:
                    st.subheader("Non-Cumulative Wins")
                    st.altair_chart(non_cumulative_wins_chart, use_container_width=True)
                with subtab_cum:
                    st.subheader("Cumulative Wins")
                    st.altair_chart(cumulative_wins_chart, use_container_width=True)

        # --- Points Tab ---
        with chart_tab_points:
            subtab_curr, subtab_trend = st.tabs(
                ["Current Standings", "Trends Over Time"]
            )
            with subtab_curr:
                st.subheader("Points per Player (Current)")
                st.altair_chart(points_chart, use_container_width=True)
            with subtab_trend:
                subtab_non_cum, subtab_cum = st.tabs(["Non-Cumulative", "Cumulative"])
                with subtab_non_cum:
                    st.subheader("Non-Cumulative Points")
                    st.altair_chart(
                        non_cumulative_points_chart, use_container_width=True
                    )
                with subtab_cum:
                    st.subheader("Cumulative Points")
                    st.altair_chart(cumulative_points_chart, use_container_width=True)

    # ------------- 4) AVG. MARGIN  -------------
    with subtab_margins:
        st.subheader("Average Margin of Victory & Defeat")

        df_margin_vic = df_filtered.groupby("Winner")["PointDiff"].mean().reset_index()
        df_margin_vic.columns = ["Player", "Avg_margin_victory"]

        df_margin_def = (
            df_filtered.groupby("Loser")["LoserPointDiff"].mean().reset_index()
        )
        df_margin_def.columns = ["Player", "Avg_margin_defeat"]

        df_margin_summary = pd.merge(
            df_margin_vic, df_margin_def, on="Player", how="outer"
        ).fillna(0)

        # Tabs for Current Standings and Trends Over Time
        margin_tabs = st.tabs(["Current Standings", "Trends Over Time"])

        with margin_tabs[0]:
            st.subheader("Current Standings: Average Margins")
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

        with margin_tabs[1]:
            st.subheader("Trends Over Time: Average Margins")
            avg_margin_victory_tab, avg_margin_defeat_tab = st.tabs(
                ["Avg. Margin of Victory", "Avg. Margin of Defeat"]
            )
            with avg_margin_victory_tab:
                # Prepare data for trends over time
                df_margin_vic = (
                    df_filtered.groupby(["date", "Winner"])["PointDiff"]
                    .mean()
                    .reset_index()
                    .rename(
                        columns={"Winner": "Player", "PointDiff": "Avg_margin_victory"}
                    )
                )

                # Trend chart for average margin of victory
                trend_chart_victory = (
                    alt.Chart(df_margin_vic)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y(
                            "Avg_margin_victory:Q", title="Average Margin of Victory"
                        ),
                        color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                        tooltip=["date:T", "Player:N", "Avg_margin_victory:Q"],
                    )
                    .properties(
                        title="Trends in Average Margin of Victory Over Time",
                        width=700,
                        height=400,
                    )
                )
                st.altair_chart(trend_chart_victory, use_container_width=True)

            with avg_margin_defeat_tab:
                # Prepare data for trends over time
                df_margin_def = (
                    df_filtered.groupby(["date", "Loser"])["LoserPointDiff"]
                    .mean()
                    .reset_index()
                    .rename(
                        columns={
                            "Loser": "Player",
                            "LoserPointDiff": "Avg_margin_defeat",
                        }
                    )
                )

                # Trend chart for average margin of defeat
                trend_chart_defeat = (
                    alt.Chart(df_margin_def)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y(
                            "Avg_margin_defeat:Q", title="Average Margin of Defeat"
                        ),
                        color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                        tooltip=["date:T", "Player:N", "Avg_margin_defeat:Q"],
                    )
                    .properties(
                        title="Trends in Average Margin of Defeat Over Time",
                        width=700,
                        height=400,
                    )
                )
                st.altair_chart(trend_chart_defeat, use_container_width=True)

    # ------------- 5) WIN/LOSS STREAKS  -------------
    with subtab_streaks:
        st.subheader("Winning and Losing Streaks")
        df_sorted = df_filtered.sort_values(["date"], ascending=True)
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
            streaks, columns=["Player", "Longest_Win_Streak", "Longest_Loss_Streak"]
        )
        streaks_df.sort_values("Longest_Win_Streak", ascending=False, inplace=True)
        st.dataframe(streaks_df, use_container_width=True)

    # ------------- 6) ENDURANCE METRICS  -------------
    with subtab_endurance:
        st.subheader("Endurance Metrics: Performance by Nth Match of Day")

        def meltdown_day_matches(df_in):
            df_in = df_in.sort_values(
                ["date", "match_number_total", "match_number_day"], ascending=True
            )

            df_p1 = df_in[
                [
                    "date",
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

            df_p2 = df_in[
                [
                    "date",
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

            df_stacked = pd.concat([df_p1, df_p2], ignore_index=True)
            df_stacked = df_stacked.sort_values(
                ["date", "player", "match_number_total", "match_number_day"]
            )
            df_stacked["MatchOfDay"] = (
                df_stacked.groupby(["date", "player"]).cumcount() + 1
            )
            return df_stacked

        df_daycount = meltdown_day_matches(df_filtered)
        df_day_agg = (
            df_daycount.groupby(["player", "MatchOfDay"])["did_win"]
            .agg(["sum", "count"])
            .reset_index()
        )
        df_day_agg["win_rate"] = df_day_agg["sum"] / df_day_agg["count"]

        # Let user select which players to show
        available_players = sorted(df_day_agg["player"].unique())
        players_for_nth_chart = st.multiselect(
            "Select which players to display in the Nth-Match-of-Day chart",
            options=available_players,
            default=available_players,
        )

        if players_for_nth_chart:
            df_day_agg_display = df_day_agg[
                df_day_agg["player"].isin(players_for_nth_chart)
            ]

            base = alt.Chart(df_day_agg_display).encode(
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

            # Actual data line with points
            lines_layer = base.mark_line(point=True)

            # Regression line
            trend_layer = (
                base.transform_regression("MatchOfDay", "win_rate", groupby=["player"])
                .mark_line(strokeDash=[4, 4])
                .encode(opacity=alt.value(0.7))
            )

            chart_match_of_day = alt.layer(lines_layer, trend_layer).properties(
                width="container", height=400
            )
            st.altair_chart(chart_match_of_day, use_container_width=True)
        else:
            st.info("No players selected for the Nth-match-of-day chart.")

        st.markdown(
            """
            This chart shows how each **selected** player performs in their 1st, 2nd, 3rd, etc. match **per day**.  
            The **solid line** is their actual data, and the **dashed line** is a linear trend line.
            """
        )

# ==========================================================
#                  HEAD-TO-HEAD ANALYSIS
# ==========================================================
with main_tab_head2head:
    st.subheader("Head-to-Head Analysis")

    # Create top-level tabs for Wins and Points
    top_level_tabs = st.tabs(["Wins", "Points"])

    # -------------------- H2H: WINS -------------------- #
    with top_level_tabs[0]:
        # Group by Winner and Loser to get win counts
        h2h_df = (
            df_filtered.groupby(["Winner", "Loser"])
            .size()
            .reset_index(name="Wins_against")
        )
        unique_players_h2h = sorted(set(h2h_df["Winner"]) | set(h2h_df["Loser"]))

        # Create a tab for each player
        player_tabs = st.tabs([f"{player} vs ..." for player in unique_players_h2h])
        for top_player, player_tab in zip(unique_players_h2h, player_tabs):
            with player_tab:
                # Identify relevant pairs
                player_pairs = []
                for winner, loser in zip(h2h_df["Winner"], h2h_df["Loser"]):
                    if top_player in [winner, loser]:
                        pair = tuple(sorted([winner, loser]))
                        if pair not in player_pairs:
                            player_pairs.append(pair)

                # Sub-tabs for each unique pair
                for winner, loser in player_pairs:
                    pair_subtab = st.tabs([f"{winner} vs {loser}"])[0]
                    with pair_subtab:
                        # Filter data for the specific pairing
                        pair_data = df_filtered[
                            (
                                (df_filtered["Winner"] == winner)
                                & (df_filtered["Loser"] == loser)
                            )
                            | (
                                (df_filtered["Winner"] == loser)
                                & (df_filtered["Loser"] == winner)
                            )
                        ]

                        # Subtabs: current standings vs trends
                        st1, st2 = st.tabs(["Current Standings", "Trends Over Time"])
                        with st1:
                            st.subheader(f"Current Standings: {winner} vs {loser}")
                            # Bar chart for head-to-head wins
                            win_counts = (
                                pair_data.groupby("Winner")
                                .size()
                                .reset_index(name="Wins")
                            )
                            win_chart = (
                                alt.Chart(win_counts)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Winner:N", title="Player"),
                                    y=alt.Y("Wins:Q", title="Number of Wins"),
                                    tooltip=["Winner:N", "Wins:Q"],
                                )
                                .properties(
                                    title=f"Head-to-Head Wins: {winner} vs {loser}",
                                    width=700,
                                    height=400,
                                )
                            )
                            st.altair_chart(win_chart, use_container_width=True)

                        with st2:
                            st.subheader(f"Trends Over Time: {winner} vs {loser}")
                            st2a, st2b = st.tabs(["Non-Cumulative", "Cumulative"])
                            with st2a:
                                st.subheader(
                                    f"Non-Cumulative Wins: {winner} vs {loser}"
                                )
                                pair_data["match_date"] = pair_data["date"]
                                non_cumulative = (
                                    pair_data.groupby(["match_date", "Winner"])
                                    .size()
                                    .reset_index(name="Wins")
                                )
                                non_cumulative_chart = (
                                    alt.Chart(non_cumulative)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("match_date:T", title="Date"),
                                        y=alt.Y("Wins:Q", title="Wins Per Match"),
                                        color=alt.Color("Winner:N", title="Player"),
                                        tooltip=["match_date:T", "Winner:N", "Wins:Q"],
                                    )
                                    .properties(
                                        title=f"Non-Cumulative Wins Over Time: {winner} vs {loser}",
                                        width=700,
                                        height=400,
                                    )
                                )
                                st.altair_chart(
                                    non_cumulative_chart, use_container_width=True
                                )

                            with st2b:
                                st.subheader(f"Cumulative Wins: {winner} vs {loser}")
                                cumulative = pair_data.copy()
                                cumulative["CumulativeWins"] = (
                                    cumulative.groupby("Winner").cumcount() + 1
                                )
                                cumulative_chart = (
                                    alt.Chart(cumulative)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("date:T", title="Date"),
                                        y=alt.Y(
                                            "CumulativeWins:Q", title="Cumulative Wins"
                                        ),
                                        color=alt.Color("Winner:N", title="Player"),
                                        tooltip=[
                                            "date:T",
                                            "Winner:N",
                                            "CumulativeWins:Q",
                                        ],
                                    )
                                    .properties(
                                        title=f"Cumulative Wins Over Time: {winner} vs {loser}",
                                        width=700,
                                        height=400,
                                    )
                                )
                                st.altair_chart(
                                    cumulative_chart, use_container_width=True
                                )

    # ------------------- H2H: POINTS -------------------- #
    with top_level_tabs[1]:
        unique_players_points = sorted(
            set(df_filtered["Player1"]) | set(df_filtered["Player2"])
        )

        # Tabs for each player
        top_tabs_points = st.tabs(
            [f"{player} vs ..." for player in unique_players_points]
        )
        for top_player, player_tab in zip(unique_players_points, top_tabs_points):
            with player_tab:
                # Identify pairs
                player_pairs = []
                for p1, p2 in zip(df_filtered["Player1"], df_filtered["Player2"]):
                    if top_player in [p1, p2]:
                        pair = tuple(sorted([p1, p2]))
                        if pair not in player_pairs:
                            player_pairs.append(pair)

                # Sub-tabs for each pair
                for p1, p2 in player_pairs:
                    subtab = st.tabs([f"{p1} vs {p2}"])[0]
                    with subtab:
                        # Filter for that pairing
                        pair_data = df_filtered[
                            (
                                (df_filtered["Player1"] == p1)
                                & (df_filtered["Player2"] == p2)
                            )
                            | (
                                (df_filtered["Player1"] == p2)
                                & (df_filtered["Player2"] == p1)
                            )
                        ]
                        combined_data = pd.concat(
                            [
                                pair_data.assign(
                                    Player=pair_data["Player1"],
                                    Points=pair_data["Score1"],
                                ),
                                pair_data.assign(
                                    Player=pair_data["Player2"],
                                    Points=pair_data["Score2"],
                                ),
                            ],
                            ignore_index=True,
                        )

                        # Subtabs: Current Standings, Trends Over Time
                        pts_tab_current, pts_tab_trends = st.tabs(
                            ["Current Standings", "Trends Over Time"]
                        )
                        with pts_tab_current:
                            st.subheader(f"Current Standings: {p1} vs {p2}")
                            total_points = (
                                combined_data.groupby("Player")["Points"]
                                .sum()
                                .reset_index()
                            )
                            points_chart = (
                                alt.Chart(total_points)
                                .mark_bar()
                                .encode(
                                    x=alt.X("Player:N", title="Player"),
                                    y=alt.Y("Points:Q", title="Total Points"),
                                    tooltip=["Player:N", "Points:Q"],
                                )
                                .properties(
                                    title=f"Total Points: {p1} vs {p2}",
                                    width=700,
                                    height=400,
                                )
                            )
                            st.altair_chart(points_chart, use_container_width=True)

                        with pts_tab_trends:
                            st.subheader(f"Trends Over Time: {p1} vs {p2}")
                            pts_subtab_non_cum, pts_subtab_cum = st.tabs(
                                ["Non-Cumulative", "Cumulative"]
                            )

                            # Non-Cumulative
                            with pts_subtab_non_cum:
                                st.subheader(f"Non-Cumulative Points: {p1} vs {p2}")
                                points_non_cumulative = (
                                    combined_data.groupby(["date", "Player"])["Points"]
                                    .sum()
                                    .reset_index()
                                )
                                non_cumulative_chart = (
                                    alt.Chart(points_non_cumulative)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("date:T", title="Date"),
                                        y=alt.Y("Points:Q", title="Points Per Match"),
                                        color=alt.Color("Player:N", title="Player"),
                                        tooltip=["date:T", "Player:N", "Points:Q"],
                                    )
                                    .properties(
                                        title=f"Non-Cumulative Points Over Time: {p1} vs {p2}",
                                        width=700,
                                        height=400,
                                    )
                                )
                                st.altair_chart(
                                    non_cumulative_chart, use_container_width=True
                                )

                            # Cumulative
                            with pts_subtab_cum:
                                st.subheader(f"Cumulative Points: {p1} vs {p2}")
                                points_cumulative = points_non_cumulative.copy()
                                points_cumulative["CumulativePoints"] = (
                                    points_cumulative.groupby("Player")[
                                        "Points"
                                    ].cumsum()
                                )

                                cumulative_chart = (
                                    alt.Chart(points_cumulative)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("date:T", title="Date"),
                                        y=alt.Y(
                                            "CumulativePoints:Q",
                                            title="Cumulative Points",
                                        ),
                                        color=alt.Color("Player:N", title="Player"),
                                        tooltip=[
                                            "date:T",
                                            "Player:N",
                                            "CumulativePoints:Q",
                                        ],
                                    )
                                    .properties(
                                        title=f"Cumulative Points Over Time: {p1} vs {p2}",
                                        width=700,
                                        height=400,
                                    )
                                )
                                st.altair_chart(
                                    cumulative_chart, use_container_width=True
                                )
