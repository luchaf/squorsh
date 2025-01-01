import streamlit as st
import pandas as pd
import altair as alt
from collections import defaultdict


# Function: Matches Over Time
def render_matches_over_time(df):
    st.subheader("Matches Over Time")
    matches_over_time = df.groupby("date").size().reset_index(name="Matches")
    chart = (
        alt.Chart(matches_over_time)
        .mark_bar()
        .encode(x="date:T", y="Matches:Q", tooltip=["date:T", "Matches:Q"])
        .properties(width="container", height=400)
    )
    st.altair_chart(chart, use_container_width=True)


# Function: Match Distribution
def render_match_distribution(df):
    st.subheader("Match Result Distribution")
    df["ResultPair"] = df.apply(
        lambda row: f"{int(max(row['Score1'], row['Score2']))}:{int(min(row['Score1'], row['Score2']))}",
        axis=1,
    )
    pair_counts = df["ResultPair"].value_counts().reset_index()
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


# Function: Legendary Matches
def render_legendary_matches(df):
    st.subheader("The Ten Most Legendary Matches")
    n_closest = 10
    df["TotalPoints"] = df["Score1"] + df["Score2"]

    # Sort by margin ascending, then total points descending
    df_closest_sorted = df.sort_values(
        ["PointDiff", "TotalPoints"], ascending=[True, False]
    )
    closest_subset = df_closest_sorted.head(n_closest)

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


# Function: Wins & Points
def render_wins_points(df):
    st.subheader("Wins & Points")

    # Wins & Points Summary
    wins_df = df.groupby("Winner").size().reset_index(name="Wins")
    points_p1 = df.groupby("Player1")["Score1"].sum().reset_index()
    points_p1.columns = ["Player", "Points"]
    points_p2 = df.groupby("Player2")["Score2"].sum().reset_index()
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
    final_summary.sort_values("Wins", ascending=False, inplace=True)

    st.dataframe(final_summary, use_container_width=True)


# Function: Average Margin
def render_avg_margin(df):
    st.subheader("Average Margin of Victory & Defeat")

    df_margin_vic = df.groupby("Winner")["PointDiff"].mean().reset_index()
    df_margin_vic.columns = ["Player", "Avg_margin_victory"]

    df_margin_def = df.groupby("Loser")["LoserPointDiff"].mean().reset_index()
    df_margin_def.columns = ["Player", "Avg_margin_defeat"]

    df_margin_summary = pd.merge(
        df_margin_vic, df_margin_def, on="Player", how="outer"
    ).fillna(0)

    st.dataframe(df_margin_summary, use_container_width=True)

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
    )
    st.altair_chart(margin_chart, use_container_width=True)


# Function: Streaks
def render_streaks(df):
    st.subheader("Winning and Losing Streaks")
    df_sorted = df.sort_values(["date"], ascending=True)
    streaks = []
    unique_players = sorted(set(df["Player1"]) | set(df["Player2"]))

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


# Function: Endurance Metrics
def render_endurance_metrics(df):
    st.subheader("Endurance Metrics: Performance by N-th Match of Day")

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
        df_stacked["MatchOfDay"] = df_stacked.groupby(["date", "player"]).cumcount() + 1
        return df_stacked

    df_daycount = meltdown_day_matches(df)
    df_day_agg = (
        df_daycount.groupby(["player", "MatchOfDay"])["did_win"]
        .agg(["sum", "count"])
        .reset_index()
    )
    df_day_agg["win_rate"] = df_day_agg["sum"] / df_day_agg["count"]

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
    else:
        st.info("No players selected for the Nth-match-of-day chart.")

    st.markdown(
        """
        This chart shows how each **selected** player performs in their 1st, 2nd, 3rd, etc. match **per day**.  
        The **solid line** is their actual data, and the **dashed line** is a linear trend line.
        """
    )


# Main Tabs
with st.tabs(["Overall Overanalysis", "Head-to-Head"])[0]:
    st.subheader("Overall Analysis")

    render_matches_over_time(df_filtered)
    render_match_distribution(df_filtered)
    render_legendary_matches(df_filtered)
    render_wins_points(df_filtered)
    render_avg_margin(df_filtered)
    render_streaks(df_filtered)
    render_endurance_metrics(df_filtered)

with st.tabs(["Overall Overanalysis", "Head-to-Head"])[1]:
    st.subheader("Head-to-Head Analysis")
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Select Player 1", options=all_players)
    with col2:
        player2 = st.selectbox("Select Player 2", options=all_players)

    if player1 and player2 and player1 != player2:
        df_h2h = df_filtered[
            ((df_filtered["Player1"] == player1) & (df_filtered["Player2"] == player2))
            | (
                (df_filtered["Player1"] == player2)
                & (df_filtered["Player2"] == player1)
            )
        ]
        if not df_h2h.empty:
            render_matches_over_time(df_h2h)
            render_match_distribution(df_h2h)
            render_legendary_matches(df_h2h)
            render_wins_points(df_h2h)
            render_avg_margin(df_h2h)
            render_streaks(df_h2h)
            render_endurance_metrics(df_h2h)
        else:
            st.write("No matches found between the selected players.")
