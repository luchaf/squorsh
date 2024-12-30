import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
from itertools import combinations
from streamlit_gsheets import GSheetsConnection

# ---- SETUP ----
conn = st.connection("gsheets", type=GSheetsConnection)
worksheet_name = "match_results"
df = conn.read(worksheet=worksheet_name)

# ---- Convert date column to datetime ----
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

# ---- Basic data cleanup ----
df["Player1"] = df["Player1"].replace("Friede", "Friedemann")
df["Player2"] = df["Player2"].replace("Friede", "Friedemann")

# ---- Derive winner/loser columns ----
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

# ---- FILTERS (Top Section) ----
st.sidebar.header("Filters")

# Date & Time Filters
days_present_in_data = sorted(set(df["day_of_week"]))
selected_days = st.sidebar.multiselect(
    "Select Day(s) of the Week to Include",
    options=days_present_in_data,
    default=days_present_in_data,
)

# Player Filters
all_players = sorted(set(df["Player1"]) | set(df["Player2"]))
selected_players = st.sidebar.multiselect(
    "Select Player(s) to Include", options=all_players, default=all_players
)

# Apply Filters
df_filtered = df[
    (df["day_of_week"].isin(selected_days))
    & ((df["Player1"].isin(selected_players)) | (df["Player2"].isin(selected_players)))
]

# ---- ORGANIZATION: TABS ----
tab_summary, tab_head_to_head = st.tabs(["Summary Metrics", "Head-to-Head"])

# =========================
#       TAB: SUMMARY
# =========================
with tab_summary:
    st.subheader("Matches Over Time")
    matches_over_time = df_filtered.groupby("date").size().reset_index(name="Matches")
    chart = (
        alt.Chart(matches_over_time)
        .mark_bar()
        .encode(x="date:T", y="Matches:Q", tooltip=["date:T", "Matches:Q"])
        .properties(width="container", height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Elo Ratings", expanded=False):
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

    with st.expander("Wins & Points", expanded=False):
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

        # Data preparation
        players_wins = final_summary_wins["Player"]
        players_points = final_summary_points["Player"]
        wins = final_summary_wins["Wins"]
        points = final_summary_points["Points"]

        # Create Altair charts
        # Chart for Wins
        wins_chart = (
            alt.Chart(final_summary_wins)
            .mark_bar(color="blue")
            .encode(
                x=alt.X("Player:N", sort=list(players_wins), title="Player"),
                y=alt.Y("Wins:Q", title="Number of Wins"),
                tooltip=["Player:N", "Wins:Q"],
            )
            .properties(title="Number of Wins by Player", width=700, height=400)
        )

        # Chart for Points
        points_chart = (
            alt.Chart(final_summary_points)
            .mark_bar(color="orange")
            .encode(
                x=alt.X("Player:N", sort=list(players_points), title="Player"),
                y=alt.Y("Points:Q", title="Total Points"),
                tooltip=["Player:N", "Points:Q"],
            )
            .properties(title="Total Points by Player", width=700, height=400)
        )

        # ---- Data Preparation for Over Time Analysis ----
        # Filtered Data: Aggregate Wins and Points by Date
        wins_over_time = (
            df_filtered.groupby(["date", "Winner"]).size().reset_index(name="Wins")
        )
        points_p1_over_time = (
            df_filtered.groupby(["date", "Player1"])["Score1"].sum().reset_index()
        )
        points_p2_over_time = (
            df_filtered.groupby(["date", "Player2"])["Score2"].sum().reset_index()
        )

        # Combine Points Over Time
        points_over_time = pd.concat(
            [
                points_p1_over_time.rename(
                    columns={"Player1": "Player", "Score1": "Points"}
                ),
                points_p2_over_time.rename(
                    columns={"Player2": "Player", "Score2": "Points"}
                ),
            ],
            ignore_index=True,
        )

        # Aggregate Points by Date and Player
        points_over_time = (
            points_over_time.groupby(["date", "Player"])["Points"].sum().reset_index()
        )

        # Aggregate Wins by Date and Player
        wins_over_time = (
            wins_over_time.groupby(["date", "Winner"])["Wins"].sum().reset_index()
        )
        wins_over_time.rename(columns={"Winner": "Player"}, inplace=True)

        # ---- ORGANIZATION: CHARTS ----
        # Chart for Wins Over Time
        wins_over_time_chart = (
            alt.Chart(wins_over_time)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Wins:Q", title="Cumulative Wins"),
                color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                tooltip=["date:T", "Player:N", "Wins:Q"],
            )
            .properties(title="Wins Development Over Time", width=700, height=400)
        )

        # Chart for Points Over Time
        points_over_time_chart = (
            alt.Chart(points_over_time)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Points:Q", title="Cumulative Points"),
                color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                tooltip=["date:T", "Player:N", "Points:Q"],
            )
            .properties(title="Points Development Over Time", width=700, height=400)
        )

        # ---- Data Preparation for Cumulative Analysis ----
        # Calculate cumulative Wins over time
        wins_over_time["CumulativeWins"] = wins_over_time.groupby("Player")[
            "Wins"
        ].cumsum()

        # Calculate cumulative Points over time
        points_over_time["CumulativePoints"] = points_over_time.groupby("Player")[
            "Points"
        ].cumsum()

        # ---- ORGANIZATION: CHARTS ----
        # Chart for Non-Cumulative Wins Over Time
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

        # Chart for Cumulative Wins Over Time
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

        # Chart for Non-Cumulative Points Over Time
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

        # Chart for Cumulative Points Over Time
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

        # ---- ORGANIZATION: STREAMLIT TABS ----
        tab1, tab2 = st.tabs(["Wins", "Points"])

        # Wins Chart Tabs
        with tab1:
            subtab1, subtab2 = st.tabs(["Current Standings", "Trends Over Time"])
            with subtab1:
                st.subheader("Current Standings: Wins per Player")
                st.altair_chart(wins_chart, use_container_width=True)
            with subtab2:
                subtab2a, subtab2b = st.tabs(["Non-Cumulative", "Cumulative"])
                with subtab2a:
                    st.subheader("Trends Over Time: Non-Cumulative Wins")
                    st.altair_chart(non_cumulative_wins_chart, use_container_width=True)
                with subtab2b:
                    st.subheader("Trends Over Time: Cumulative Wins")
                    st.altair_chart(cumulative_wins_chart, use_container_width=True)

        # Points Chart Tabs
        with tab2:
            subtab1, subtab2 = st.tabs(["Current Standings", "Trends Over Time"])
            with subtab1:
                st.subheader("Current Standings: Points per Player")
                st.altair_chart(points_chart, use_container_width=True)
            with subtab2:
                subtab2a, subtab2b = st.tabs(["Non-Cumulative", "Cumulative"])
                with subtab2a:
                    st.subheader("Trends Over Time: Non-Cumulative Points")
                    st.altair_chart(
                        non_cumulative_points_chart, use_container_width=True
                    )
                with subtab2b:
                    st.subheader("Trends Over Time: Cumulative Points")
                    st.altair_chart(cumulative_points_chart, use_container_width=True)

        # ----- Avg Margin of Victory & Defeat (Per Player) -----

    with st.expander("Average Margin of Victory & Defeat2", expanded=False):
        st.subheader("Average Margin of Victory & Defeat")

        df_margin_vic = (
            df_filtered.groupby(["date", "Winner"])["PointDiff"].mean().reset_index()
        )
        df_margin_vic.columns = ["date", "Player", "Avg_margin_victory"]

        df_margin_def = (
            df_filtered.groupby(["date", "Loser"])["LoserPointDiff"]
            .mean()
            .reset_index()
        )
        df_margin_def.columns = ["date", "Player", "Avg_margin_defeat"]

        df_margin_summary = pd.concat([df_margin_vic, df_margin_def], ignore_index=True)

        # Tabs for Current Standings and Trends Over Time
        margin_tabs = st.tabs(["Current Standings", "Trends Over Time"])

        with margin_tabs[0]:
            st.subheader("Current Standings: Average Margins")

            summary_vic = (
                df_margin_vic.groupby("Player")["Avg_margin_victory"]
                .mean()
                .reset_index()
            )
            summary_def = (
                df_margin_def.groupby("Player")["Avg_margin_defeat"]
                .mean()
                .reset_index()
            )

            margin_summary = pd.merge(
                summary_vic, summary_def, on="Player", how="outer"
            ).fillna(0)

            margin_chart = (
                alt.Chart(margin_summary)
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
                .properties(
                    title="Average Margins for Victory and Defeat",
                    width=700,
                    height=400,
                )
            )

            st.altair_chart(margin_chart, use_container_width=True)

        with margin_tabs[1]:
            st.subheader("Trends Over Time: Average Margins")

            trend_chart = (
                alt.Chart(df_margin_summary)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("value:Q", title="Average Margin"),
                    color=alt.Color("variable:N", title="Metric"),
                    detail="Player:N",
                    tooltip=["date:T", "Player:N", "variable:N", "value:Q"],
                )
                .facet(row=alt.Row("Player:N", title="Player"), spacing=20)
                .properties(
                    title="Trends in Average Margins Over Time", width=700, height=100
                )
            )

            st.altair_chart(trend_chart, use_container_width=True)

        with margin_tabs[1]:
            st.subheader("Trends Over Time: Average Margins")

            trend_chart = (
                alt.Chart(df_margin_summary)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("value:Q", title="Average Margin"),
                    color=alt.Color("variable:N", title="Metric"),
                    detail="Player:N",
                    tooltip=["date:T", "Player:N", "variable:N", "value:Q"],
                )
                .facet(row=alt.Row("Player:N", title="Player"), spacing=20)
                .properties(
                    title="Trends in Average Margins Over Time", width=700, height=100
                )
            )

            st.altair_chart(trend_chart, use_container_width=True)

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
        df_margin_summary.sort_values("Player", inplace=True)

        # Tabs for Current Standings and Trends Over Time
        margin_tabs = st.tabs(["Current Standings", "Trends Over Time"])

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
                .properties(
                    title="Average Margins for Victory and Defeat",
                    width=700,
                    height=400,
                )
            )

            st.altair_chart(margin_chart, use_container_width=True)

        with margin_tabs[1]:
            st.subheader("Trends Over Time: Average Margins")

            # Prepare data for trends over time
            df_margin_over_time = pd.concat(
                [
                    df_filtered.groupby(["date", "Winner"])["PointDiff"]
                    .mean()
                    .reset_index()
                    .rename(
                        columns={"Winner": "Player", "PointDiff": "Avg_margin_victory"}
                    ),
                    df_filtered.groupby(["date", "Loser"])["LoserPointDiff"]
                    .mean()
                    .reset_index()
                    .rename(
                        columns={
                            "Loser": "Player",
                            "LoserPointDiff": "Avg_margin_defeat",
                        }
                    ),
                ]
            )

            df_margin_over_time = pd.melt(
                df_margin_over_time,
                id_vars=["date", "Player"],
                value_vars=["Avg_margin_victory", "Avg_margin_defeat"],
                var_name="Metric",
                value_name="Value",
            )

            trend_chart = (
                alt.Chart(df_margin_over_time)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Value:Q", title="Average Margin"),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=["date:T", "Player:N", "Metric:N", "Value:Q"],
                )
                .properties(
                    title="Trends in Average Margins Over Time", width=700, height=400
                )
            )

            st.altair_chart(trend_chart, use_container_width=True)

    with st.expander("Average Margin of Victory & Defeat", expanded=False):
        df_margin_vic = df_filtered.groupby("Winner")["PointDiff"].mean().reset_index()
        df_margin_vic.columns = ["Player", "Avg_margin_victory"]

        df_margin_def = (
            df_filtered.groupby("Loser")["LoserPointDiff"].mean().reset_index()
        )
        df_margin_def.columns = ["Player", "Avg_margin_defeat"]

        df_margin_summary = pd.merge(
            df_margin_vic, df_margin_def, on="Player", how="outer"
        ).fillna(0)
        df_margin_summary.sort_values("Player", inplace=True)

        st.dataframe(
            df_margin_summary.style.format(
                {"Avg_margin_victory": "{:.2f}", "Avg_margin_defeat": "{:.2f}"}
            ),
            use_container_width=True,
        )

    with st.expander("Winning and Losing Streaks", expanded=False):
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

    with st.expander("Endurance Metrics", expanded=False):

        st.subheader("Performance by Nth Match of Day")

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
            ]
            df_p1 = df_p1.rename(
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
            ]
            df_p2 = df_p2.rename(
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

        # Let user select which players to show in the chart
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

            # 1) Actual data line with points
            lines_layer = base.mark_line(point=True)

            # 2) Regression line
            trend_layer = (
                base.transform_regression("MatchOfDay", "win_rate", groupby=["player"])
                .mark_line(strokeDash=[4, 4])
                .encode(opacity=alt.value(0.7))
            )

            chart_match_of_day = alt.layer(lines_layer, trend_layer).properties(
                width="container", height=400
            )

            st.altair_chart(chart_match_of_day, use_container_width=True)

            st.markdown("**Table**: Win Rate by Nth Match of Day (Filtered)")
            st.dataframe(
                df_day_agg_display[["player", "MatchOfDay", "sum", "count", "win_rate"]]
                .sort_values(["player", "MatchOfDay"])
                .reset_index(drop=True)
                .style.format({"win_rate": "{:.2f}"}),
                use_container_width=True,
            )
        else:
            st.info("No players selected for the Nth-match-of-day chart.")

        st.markdown(
            """
        This chart & table show how each **selected** player performs in their 1st, 2nd, 3rd, etc. match **per day**.  
        The **solid line** is their actual data, and the **dashed line** is a linear trend line.  
        """
        )

    with st.expander("Match Result Distribution", expanded=False):
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

    with st.expander("List of the ten most legendary matches", expanded=False):

        st.subheader("Closest Matches (Filtered)")

        n_closest = 10
        df_filtered["TotalPoints"] = df_filtered["Score1"] + df_filtered["Score2"]

        # Sort by margin ascending, then by total points descending
        df_closest_sorted = df_filtered.sort_values(
            ["PointDiff", "TotalPoints"], ascending=[True, False]
        )
        closest_subset = df_closest_sorted.head(n_closest)
        closest_subset["date"] = pd.to_datetime(
            df["date"]
        )  # Ensure the column is in datetime format
        closest_subset["date"] = closest_subset[
            "date"
        ].dt.date  # Extract only the date part

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

with tab_head_to_head:
    st.subheader("Head-to-Head Analysis")

    # Create top-level tabs for Wins and Points
    top_level_tabs = st.tabs(["Wins", "Points"])

    # ====== WINS TAB ======
    with top_level_tabs[0]:
        # Group by Winner and Loser to get win counts
        h2h_df = (
            df_filtered.groupby(["Winner", "Loser"])
            .size()
            .reset_index(name="Wins_against")
        )

        # Create list of unique players for top-level tabs
        unique_players = sorted(set(h2h_df["Winner"]) | set(h2h_df["Loser"]))

        # Create top-level tabs for each player
        top_tabs = st.tabs([f"{player} vs ..." for player in unique_players])

        for i, (top_player, top_tab) in enumerate(zip(unique_players, top_tabs)):
            with top_tab:

                # Filter matchups to ensure each pairing is shown only once
                player_pairs = [
                    (winner, loser)
                    for winner, loser in zip(h2h_df["Winner"], h2h_df["Loser"])
                    if top_player == winner or top_player == loser
                ]
                filtered_pairs = []
                seen_pairs = set()

                for winner, loser in player_pairs:
                    pair = tuple(sorted([winner, loser]))  # Sort to avoid duplicates
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        filtered_pairs.append((winner, loser))

                # Create sub-tabs for each player pairing
                for winner, loser in filtered_pairs:
                    pair_tab = st.tabs([f"{winner} vs {loser}"])[0]
                    with pair_tab:

                        # Filter data for the specific player pairing
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

                        # ---- Sub-tabs for the pairing ----
                        subtab_current, subtab_trends = st.tabs(
                            ["Current Standings", "Trends Over Time"]
                        )

                        # ---- Current Standings ----
                        with subtab_current:
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

                        # ---- Trends Over Time ----
                        with subtab_trends:
                            st.subheader(f"Trends Over Time: {winner} vs {loser}")

                            subtab_non_cumulative, subtab_cumulative = st.tabs(
                                ["Non-Cumulative", "Cumulative"]
                            )

                            # ---- Non-Cumulative Trends ----
                            with subtab_non_cumulative:
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

                            # ---- Cumulative Trends ----
                            with subtab_cumulative:
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

    # ====== POINTS TAB ======
    with top_level_tabs[1]:
        # Create list of unique players for top-level tabs
        unique_players = sorted(
            set(df_filtered["Player1"]) | set(df_filtered["Player2"])
        )

        # Create top-level tabs for each player
        top_tabs = st.tabs([f"{player} vs ..." for player in unique_players])

        for top_player, top_tab in zip(unique_players, top_tabs):
            with top_tab:

                # Filter matchups to ensure each pairing is shown only once
                player_pairs = [
                    (player1, player2)
                    for player1, player2 in zip(
                        df_filtered["Player1"], df_filtered["Player2"]
                    )
                    if top_player == player1 or top_player == player2
                ]
                filtered_pairs = []
                seen_pairs = set()

                for player1, player2 in player_pairs:
                    pair = tuple(sorted([player1, player2]))  # Sort to avoid duplicates
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        filtered_pairs.append((player1, player2))

                # Create sub-tabs for each player pairing
                for player1, player2 in filtered_pairs:
                    pair_tab = st.tabs([f"{player1} vs {player2}"])[0]
                    with pair_tab:

                        # Combine data for the specific player pairing
                        pair_data = df_filtered[
                            (
                                (df_filtered["Player1"] == player1)
                                & (df_filtered["Player2"] == player2)
                            )
                            | (
                                (df_filtered["Player1"] == player2)
                                & (df_filtered["Player2"] == player1)
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

                        # ---- Sub-tabs for the pairing ----
                        subtab_current, subtab_trends = st.tabs(
                            ["Current Standings", "Trends Over Time"]
                        )

                        # ---- Current Standings ----
                        with subtab_current:
                            st.subheader(f"Current Standings: {player1} vs {player2}")

                            # Points chart for head-to-head
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
                                    title=f"Total Points: {player1} vs {player2}",
                                    width=700,
                                    height=400,
                                )
                            )

                            st.altair_chart(points_chart, use_container_width=True)

                        # ---- Trends Over Time ----
                        with subtab_trends:
                            st.subheader(f"Trends Over Time: {player1} vs {player2}")

                            subtab_non_cumulative, subtab_cumulative = st.tabs(
                                ["Non-Cumulative", "Cumulative"]
                            )

                            # ---- Non-Cumulative Trends ----
                            with subtab_non_cumulative:
                                st.subheader(
                                    f"Non-Cumulative Points: {player1} vs {player2}"
                                )

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
                                        title=f"Non-Cumulative Points Over Time: {player1} vs {player2}",
                                        width=700,
                                        height=400,
                                    )
                                )

                                st.altair_chart(
                                    non_cumulative_chart, use_container_width=True
                                )

                            # ---- Cumulative Trends ----
                            with subtab_cumulative:
                                st.subheader(
                                    f"Cumulative Points: {player1} vs {player2}"
                                )

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
                                        title=f"Cumulative Points Over Time: {player1} vs {player2}",
                                        width=700,
                                        height=400,
                                    )
                                )

                                st.altair_chart(
                                    cumulative_chart, use_container_width=True
                                )
