import pandas as pd
import altair as alt
import streamlit as st

from rating_utils import (
    generate_elo_ratings,
    generate_glicko2_ratings,
    generate_trueskill_ratings,
)

from visualization_utils import (
    chart_match_distribution,
    chart_matches_over_time,
    chart_match_intensity_over_time,
    chart_win_rate_by_day_of_week,
    chart_wins_barchart,
    chart_points_barchart,
    chart_wins_over_time,
    chart_points_over_time,
    chart_streaks_over_time,
)
from dataframe_utils import (
    generate_wins_points_summary,
    meltdown_day_matches,
    get_legendary_matches,
    compute_streak_timeseries,
)


def display_match_stats(df_filtered: pd.DataFrame):
    """
    Section: "Match Stats"
    Subtabs:
      1) "Matches Over Time"
      2) "Match Distribution"
      3) "Match Intensity"
      4) "Day-of-Week Performance"
    """
    (
        match_time_tab,
        match_dist_tab,
        intensity_tab,
        dayofweek_tab,
    ) = st.tabs(
        [
            "Matches Over Time",
            "Match Distribution",
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

    with intensity_tab:
        st.subheader("Match Intensity (Average Total Points Over Time)")
        st.altair_chart(
            chart_match_intensity_over_time(df_filtered), use_container_width=True
        )

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


def generate_analysis_content(df_filtered: pd.DataFrame, include_ratings: bool):
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
    if not include_ratings:
        # If we skip the rating systems entirely
        list_of_tabs.remove("Ratings")

    tabs = st.tabs(list_of_tabs)
    idx = 0

    # 1) MATCH STATS
    with tabs[idx]:
        display_match_stats(df_filtered)
    idx += 1

    # 2) RATINGS (Optional)
    if include_ratings:
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
