import pandas as pd
import altair as alt
import streamlit as st
from typing import Dict

from info_components import create_section_info, SECTION_EXPLANATIONS

from rating_utils import (
    generate_glicko2_ratings_over_time,
    generate_elo_ratings_over_time,
    generate_trueskill_ratings_over_time,
    plot_ratings_over_time,
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
    chart_win_rate_barchart,
    chart_win_rate_over_time,
    chart_win_rate_by_year,
    chart_win_rate_by_month_of_year,
)
from dataframe_utils import (
    generate_wins_points_summary,
    meltdown_day_matches,
    get_legendary_matches,
    compute_streak_timeseries,
)

# Import advanced metrics modules
from advanced_metrics import (
    calculate_clutch_performance,
    analyze_comebacks,
    calculate_dominance_score,
    calculate_consistency_rating,
    calculate_momentum_metrics,
    calculate_fatigue_analysis,
    calculate_nemesis_analysis,
    calculate_improvement_velocity,
    calculate_pressure_performance,
    calculate_optimal_rest_period
)

from predictive_analytics import (
    calculate_head_to_head_probability,
    predict_score_distribution,
    calculate_upset_potential,
    calculate_performance_trajectory
)

from enhanced_visualizations import (
    create_player_comparison_radar,
    create_win_rate_opponent_heatmap,
    create_bubble_performance_chart,
    create_sparklines_dataframe,
    create_momentum_indicator
)


@st.cache_data
def calculate_all_advanced_metrics(df: pd.DataFrame) -> Dict:
    """Calculate all advanced metrics with caching for performance"""
    metrics = {}
    
    # Calculate Elo ratings
    elo_df = generate_elo_ratings_over_time(df)
    latest_elo = elo_df.groupby("Player").last()["Elo Rating"].to_dict()
    
    # Advanced metrics
    metrics['clutch'] = calculate_clutch_performance(df)
    metrics['comebacks'] = analyze_comebacks(df)
    metrics['dominance'] = calculate_dominance_score(df)
    metrics['consistency'] = calculate_consistency_rating(df)
    metrics['momentum'] = calculate_momentum_metrics(df)
    metrics['pressure'] = calculate_pressure_performance(df)
    metrics['fatigue'] = calculate_fatigue_analysis(df)
    metrics['nemesis'] = calculate_nemesis_analysis(df)
    metrics['improvement'] = calculate_improvement_velocity(df, elo_df)
    metrics['optimal_rest'] = calculate_optimal_rest_period(df)
    metrics['elo_ratings'] = latest_elo
    metrics['elo_history'] = elo_df
    
    return metrics


def display_match_stats(df_filtered: pd.DataFrame):
    """
    Section: "Match Stats"
    Subtabs:
      1) "Matches Over Time"
      2) "Match Distribution"
      3) "Match Intensity"
      4) "Time-Based Performance"
    """
    # Add info box explanation
    create_section_info(
        "Match Stats", 
        SECTION_EXPLANATIONS["Match Stats"]["description"],
        SECTION_EXPLANATIONS["Match Stats"]["metrics"]
    )
    
    (
        match_time_tab,
        match_dist_tab,
        intensity_tab,
        time_based_tab,
    ) = st.tabs(
        [
            "Matches Over Time",
            "Match Distribution",
            "Match Intensity",
            "Time-Based Performance",
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

    with time_based_tab:
        st.subheader("Time-Based Performance")
        dayofweek_tab, monthofyear_tab, year_tab = st.tabs(
            ["Day-of-Week Performance", "Month-of-Year Performance", "Year Performance"]
        )

        with dayofweek_tab:
            st.subheader("Detailed Win Rate by Day of Week")
            st.altair_chart(
                chart_win_rate_by_day_of_week(df_filtered), use_container_width=True
            )
            st.write(
                "Hover over the chart to see each player's matches, wins, and computed win rate by day of the week."
            )

        with monthofyear_tab:
            st.subheader("Detailed Win Rate by Month of Year")
            st.altair_chart(
                chart_win_rate_by_month_of_year(df_filtered), use_container_width=True
            )
            st.write(
                "Hover over the chart to see each player's matches, wins, and computed win rate by month of the year."
            )

        with year_tab:
            st.subheader("Detailed Win Rate by Year")
            st.altair_chart(
                chart_win_rate_by_year(df_filtered), use_container_width=True
            )
            st.write(
                "Hover over the chart to see each player's matches, wins, and computed win rate by year."
            )


def display_ratings_tabs(df_filtered: pd.DataFrame):
    # Add info box explanation
    create_section_info(
        "Ratings", 
        SECTION_EXPLANATIONS["Ratings"]["description"],
        SECTION_EXPLANATIONS["Ratings"]["metrics"]
    )
    
    glicko_tab, elo_tab, trueskill_tab = st.tabs(["Glicko2", "Elo", "TrueSkill"])

    with glicko_tab:
        static_tab, dynamic_tab = st.tabs(["Static Table", "Rating Over Time"])
        with static_tab:
            glicko_df = generate_glicko2_ratings_over_time(df_filtered)
            latest_glicko_df = (
                glicko_df.groupby("Player")
                .last()
                .reset_index()
                .sort_values("Glicko2 Rating", ascending=False)
                .drop(columns=["date"])
            )
            st.dataframe(latest_glicko_df, use_container_width=True)
        with dynamic_tab:
            st.altair_chart(
                plot_ratings_over_time(
                    glicko_df, "Glicko2 Rating", "Glicko2 Rating Over Time"
                ),
                use_container_width=True,
            )

    with elo_tab:
        static_tab, dynamic_tab = st.tabs(["Static Table", "Rating Over Time"])
        with static_tab:
            elo_df = generate_elo_ratings_over_time(df_filtered)
            latest_elo_df = (
                elo_df.groupby("Player")
                .last()
                .reset_index()
                .sort_values("Elo Rating", ascending=False)
                .drop(columns=["date"])
            )
            st.dataframe(latest_elo_df, use_container_width=True)
        with dynamic_tab:
            st.altair_chart(
                plot_ratings_over_time(elo_df, "Elo Rating", "Elo Rating Over Time"),
                use_container_width=True,
            )

    with trueskill_tab:
        static_tab, dynamic_tab = st.tabs(["Static Table", "Rating Over Time"])
        with static_tab:
            ts_df = generate_trueskill_ratings_over_time(df_filtered)
            latest_ts_df = (
                ts_df.groupby("Player")
                .last()
                .reset_index()
                .sort_values("TrueSkill Rating", ascending=False)
                .drop(columns=["date"])
            )
            st.dataframe(latest_ts_df, use_container_width=True)
        with dynamic_tab:
            st.altair_chart(
                plot_ratings_over_time(
                    ts_df, "TrueSkill Rating", "TrueSkill Rating Over Time"
                ),
                use_container_width=True,
            )


def display_wins_and_points(df_filtered: pd.DataFrame):
    """
    Section: "Wins & Points"
    Subtabs: "Win Rate", "Wins", "Points", "Match Day Winner"
    Each has "Current Standings" and "Trends Over Time"
    """
    st.subheader("Wins & Points")
    
    # Add info box explanation
    create_section_info(
        "Wins & Points", 
        SECTION_EXPLANATIONS["Wins & Points"]["description"],
        SECTION_EXPLANATIONS["Wins & Points"]["metrics"]
    )

    # Summaries
    final_summary = generate_wins_points_summary(df_filtered)
    final_summary_wins = final_summary.copy()
    final_summary_points = final_summary.copy()
    final_summary_wins.sort_values(by="Wins", ascending=False, inplace=True)
    final_summary_points.sort_values(by="Points", ascending=False, inplace=True)

    # Calculate Match Day Winners
    match_day_winners = (
        df_filtered.groupby(["date", "Winner"]).size().reset_index(name="Wins")
    )

    # Identify the player with the most wins per day
    daily_top_winner = (
        match_day_winners.groupby("date")
        .apply(lambda x: x.loc[x["Wins"].idxmax()])
        .reset_index(drop=True)
    )

    # Count how many distinct days each player has won
    match_day_winner_counts = (
        daily_top_winner.groupby("Winner").size().reset_index(name="MatchDaysWon")
    )
    match_day_winner_counts.sort_values(
        by="MatchDaysWon", ascending=False, inplace=True
    )

    # --- Visualization: Bar Chart for Current Standings ---
    match_day_winner_chart = (
        alt.Chart(match_day_winner_counts)
        .mark_bar()
        .encode(
            x=alt.X("Winner:N", title="Player", sort="-y"),
            y=alt.Y("MatchDaysWon:Q", title="Match Days Won"),
            tooltip=["Winner:N", "MatchDaysWon:Q"],
        )
        .properties(title="Match Days Won by Each Player", height=400)
    )

    # --- Non-Cumulative Match Day Wins Over Time ---
    non_cum_match_days = (
        daily_top_winner.groupby(["date", "Winner"])
        .size()
        .reset_index(name="MatchDayWins")
    )

    non_cum_chart = (
        alt.Chart(non_cum_match_days)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("MatchDayWins:Q", title="Match Day Wins"),
            color=alt.Color("Winner:N", title="Player"),
            tooltip=["date:T", "Winner:N", "MatchDayWins:Q"],
        )
        .properties(title="Non-Cumulative Match Day Wins Over Time", height=400)
    )

    # --- Cumulative Match Day Wins Over Time ---
    cumulative_match_days = (
        non_cum_match_days.groupby("Winner")
        .apply(
            lambda x: x.sort_values("date").assign(
                CumulativeWins=x["MatchDayWins"].cumsum()
            )
        )
        .reset_index(drop=True)
    )

    cum_chart = (
        alt.Chart(cumulative_match_days)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("CumulativeWins:Q", title="Cumulative Match Day Wins"),
            color=alt.Color("Winner:N", title="Player"),
            tooltip=["date:T", "Winner:N", "CumulativeWins:Q"],
        )
        .properties(title="Cumulative Match Day Wins Over Time", height=400)
    )

    # Charts
    wins_chart = chart_wins_barchart(final_summary_wins)
    points_chart = chart_points_barchart(final_summary_points)
    win_rate_chart = chart_win_rate_barchart(final_summary)
    non_cum_wins, cum_wins = chart_wins_over_time(df_filtered)
    non_cum_points, cum_points = chart_points_over_time(df_filtered)
    non_cum_win_rate, cum_win_rate = chart_win_rate_over_time(df_filtered)

    chart_tab_match_day_winner, chart_tab_win_rate, chart_tab_wins, chart_tab_points = (
        st.tabs(["Match Day Wins", "Total Win Rate", "Total Wins", "Total Points"])
    )

    # --- Match Day Winner Tab ---
    with chart_tab_match_day_winner:
        subtab_curr, subtab_trend = st.tabs(["Current Standings", "Trends Over Time"])

        # --- Current Standings ---
        with subtab_curr:
            st.subheader("Match Days Won per Player (Current)")
            st.altair_chart(match_day_winner_chart, use_container_width=True)

        # --- Trends Over Time with Non-Cumulative and Cumulative Tabs ---
        with subtab_trend:
            subtab_non_cum, subtab_cum = st.tabs(["Non-Cumulative", "Cumulative"])

            # --- Non-Cumulative Tab ---
            with subtab_non_cum:
                st.subheader("Non-Cumulative Match Day Wins Over Time")
                st.altair_chart(non_cum_chart, use_container_width=True)

            # --- Cumulative Tab ---
            with subtab_cum:
                st.subheader("Cumulative Match Day Wins Over Time")
                st.altair_chart(cum_chart, use_container_width=True)

    # --- Win Rate Tab ---
    with chart_tab_win_rate:
        subtab_curr, subtab_trend = st.tabs(["Current Standings", "Trends Over Time"])
        with subtab_curr:
            st.subheader("Win Rate per Player (Current)")
            st.altair_chart(win_rate_chart, use_container_width=True)
        with subtab_trend:
            subtab_non_cum, subtab_cum = st.tabs(["Non-Cumulative", "Cumulative"])
            with subtab_non_cum:
                st.subheader("Daily Win Rate Over Time")
                st.altair_chart(non_cum_win_rate, use_container_width=True)
            with subtab_cum:
                st.subheader("Cumulative Win Rate Over Time")
                st.altair_chart(cum_win_rate, use_container_width=True)

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
    
    # Add info box explanation
    create_section_info(
        "Avg. Margin", 
        SECTION_EXPLANATIONS["Avg. Margin"]["description"],
        SECTION_EXPLANATIONS["Avg. Margin"]["metrics"]
    )

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
            selection = alt.selection_multi(fields=["Player"], bind="legend")
            victory_chart = (
                alt.Chart(df_margin_vic2)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Avg_margin_victory:Q", title="Average Margin of Victory"),
                    color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                    tooltip=["date:T", "Player:N", "Avg_margin_victory:Q"],
                    opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
                )
                .properties(
                    title="Trends in Average Margin of Victory Over Time",
                    width=700,
                    height=400,
                )
                .add_selection(selection)
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
            selection = alt.selection_multi(fields=["Player"], bind="legend")
            defeat_chart = (
                alt.Chart(df_margin_def2)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Avg_margin_defeat:Q", title="Average Margin of Defeat"),
                    color=alt.Color("Player:N", legend=alt.Legend(title="Player")),
                    tooltip=["date:T", "Player:N", "Avg_margin_defeat:Q"],
                    opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
                )
                .properties(
                    title="Trends in Average Margin of Defeat Over Time",
                    width=700,
                    height=400,
                )
                .add_selection(selection)
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
    
    # Add info box explanation
    create_section_info(
        "Win/Loss Streaks", 
        SECTION_EXPLANATIONS["Win/Loss Streaks"]["description"],
        SECTION_EXPLANATIONS["Win/Loss Streaks"]["metrics"]
    )

    tabs_streaks = st.tabs(
        ["Longest Streaks (Overall)", "Win & Loss Streaks Over Time"]
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
      2) "Nerves of Steel" (filter: 11:9 or 9:11)
      3) "Nerves of Adamantium" (filter: >= 12:10 or >= 10:12)
    """
    # Add info box explanation
    create_section_info(
        "Endurance and Grit", 
        SECTION_EXPLANATIONS["Endurance and Grit"]["description"],
        SECTION_EXPLANATIONS["Endurance and Grit"]["metrics"]
    )
    
    endurance_tabs = st.tabs(
        ["N-th Match of the Day", "Nerves of Steel", "Nerves of Adamantium"]
    )
    df_backup = df_filtered.copy()

    # --- 1) N-th Match of the Day ---
    with endurance_tabs[0]:
        st.subheader("Endurance Metrics: Performance by N-th Match of Day")
        df_daycount = meltdown_day_matches(df_filtered)

        import numpy as np
        import pandas as pd
        import statsmodels.api as sm

        # Function to calculate weighted regression
        def calculate_weighted_regression(df, x_col, y_col, weight_col, group_col):
            regression_results = []

            for player, group in df.groupby(group_col):
                X = group[x_col]
                y = group[y_col]
                weights = group[weight_col]
                X = sm.add_constant(X)  # Add constant for the intercept

                # Weighted least squares regression
                model = sm.WLS(y, X, weights=weights)
                results = model.fit()

                # Create regression line
                predicted = results.predict(X)
                group["regression"] = predicted
                regression_results.append(group)

            return pd.concat(regression_results)

        # Prepare the data
        df_day_agg = (
            df_daycount.groupby(["player", "MatchOfDay"])["did_win"]
            .agg(["sum", "count"])
            .reset_index()
        )
        df_day_agg["win_rate"] = df_day_agg["sum"] / df_day_agg["count"]

        # Calculate weighted regression
        df_with_regression = calculate_weighted_regression(
            df_day_agg,
            x_col="MatchOfDay",
            y_col="win_rate",
            weight_col="count",
            group_col="player",
        )

        selection = alt.selection_multi(fields=["player"], bind="legend")

        # Plot in Altair
        base = alt.Chart(df_with_regression).encode(
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
            opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        )

        lines_layer = base.mark_line(point=True)
        trend_layer = (
            alt.Chart(df_with_regression)
            .mark_line(strokeDash=[4, 4])
            .encode(
                x=alt.X("MatchOfDay:Q"),
                y=alt.Y("regression:Q", title="Weighted Regression"),
                color=alt.Color("player:N", title="Player"),
                opacity=alt.condition(selection, alt.value(0.7), alt.value(0.1)),
            )
        )

        chart_match_of_day = (
            alt.layer(lines_layer, trend_layer)
            .properties(width="container", height=400)
            .add_selection(selection)
        )
        st.altair_chart(chart_match_of_day, use_container_width=True)

        st.markdown(
            """
            This chart shows how each **selected** player performs in their 1st, 2nd, 3rd, etc. match **per day**.  
            The **solid line** is their actual data, and the **dashed line** is a weighted linear trend line.
            """
        )

    # --- 2) Nerves of Steel: 11:9 or 9:11 ---
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

    # --- 3) Nerves of Adamantium: >=12:10 or >=10:12 ---
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
    
    # Add info box explanation
    create_section_info(
        "Records & Leaderboards", 
        SECTION_EXPLANATIONS["Records & Leaderboards"]["description"],
        SECTION_EXPLANATIONS["Records & Leaderboards"]["metrics"]
    )

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


def display_performance_overview(df_filtered: pd.DataFrame, metrics: Dict):
    """Display Performance Overview with key metrics, sparklines, and bubble chart"""
    st.header("Performance Overview")
    
    # Add info box explanation
    create_section_info(
        "Performance Overview", 
        SECTION_EXPLANATIONS["Performance Overview"]["description"],
        SECTION_EXPLANATIONS["Performance Overview"]["metrics"]
    )
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(df_filtered))
        players = set(df_filtered["Player1"]) | set(df_filtered["Player2"])
        st.metric("Active Players", len(players))
    
    with col2:
        if not df_filtered.empty:
            winner_counts = df_filtered.groupby("Winner").size()
            if not winner_counts.empty:
                most_wins = winner_counts.idxmax()
                st.metric("Most Wins", most_wins)
                st.metric("Win Count", winner_counts.max())
            else:
                st.metric("Most Wins", "N/A")
                st.metric("Win Count", 0)
        else:
            st.metric("Most Wins", "N/A")
            st.metric("Win Count", 0)
    
    with col3:
        if metrics['elo_ratings']:
            highest_rated = max(metrics['elo_ratings'].items(), key=lambda x: x[1])
            st.metric("Highest Rated", highest_rated[0])
            st.metric("Elo Rating", f"{highest_rated[1]:.0f}")
        else:
            st.metric("Highest Rated", "N/A")
            st.metric("Elo Rating", "N/A")
    
    with col4:
        if not metrics['momentum'].empty:
            hottest_player = metrics['momentum'].iloc[0]
            st.metric("Hottest Player", hottest_player['Player'])
            st.metric("Momentum", f"{hottest_player['Momentum Score']:.1f}")
        else:
            st.metric("Hottest Player", "N/A")
            st.metric("Momentum", "0.0")
    
    # Sparklines and recent form
    st.subheader("📈 Recent Form Tracker")
    sparklines_df = create_sparklines_dataframe(df_filtered)
    
    # Display with custom styling
    for _, player_row in sparklines_df.iterrows():
        col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
        with col1:
            st.write(f"**{player_row['Player']}**")
        with col2:
            st.write(player_row['Recent Form'])
        with col3:
            st.write(f"{player_row['Trend']} {player_row['Last 10']}")
        with col4:
            momentum = metrics['momentum'][metrics['momentum']['Player'] == player_row['Player']]
            if not momentum.empty:
                st.write(momentum.iloc[0]['Hot Streak'])
    
    # Performance bubble chart
    st.subheader("🎯 Performance Overview")
    bubble_chart = create_bubble_performance_chart(df_filtered)
    st.altair_chart(bubble_chart, use_container_width=True)


def display_predictive_analytics(df_filtered: pd.DataFrame, metrics: Dict):
    """Display Predictive Analytics section"""
    st.subheader("🔮 Predictive Analytics & Match Forecasting")
    
    # Add info box explanation
    create_section_info(
        "Predictive Analytics", 
        SECTION_EXPLANATIONS["Predictive Analytics"]["description"],
        SECTION_EXPLANATIONS["Predictive Analytics"]["metrics"]
    )
    
    all_players = sorted(set(df_filtered["Player1"]) | set(df_filtered["Player2"]))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Match Predictor")
        pred_p1 = st.selectbox("Player 1", all_players, key="pred_p1_analysis")
        pred_p2 = st.selectbox("Player 2", [p for p in all_players if p != pred_p1], key="pred_p2_analysis")
        
        if st.button("Calculate Match Probability", key="calc_prob_analysis"):
            prob_data = calculate_head_to_head_probability(
                df_filtered, pred_p1, pred_p2, metrics['elo_ratings']
            )
            
            # Display probability bars
            st.metric(f"{pred_p1} Win Probability", 
                     f"{prob_data['player1_probability']*100:.1f}%")
            st.metric(f"{pred_p2} Win Probability", 
                     f"{prob_data['player2_probability']*100:.1f}%")
            
            # Confidence meter
            st.progress(prob_data['confidence'] / 100)
            st.caption(f"Prediction Confidence: {prob_data['confidence']:.0f}%")
            
            # Factors breakdown
            st.write("**Contributing Factors:**")
            factors_df = pd.DataFrame([prob_data['factors']]).T
            factors_df.columns = ['Probability']
            factors_df.index = ['Historical H2H', 'Recent H2H', 'Elo Rating', 'Current Form']
            st.dataframe(factors_df.style.format("{:.2%}"))
    
    with col2:
        st.subheader("Score Predictor")
        if 'pred_p1' in locals() and 'pred_p2' in locals():
            score_pred = predict_score_distribution(df_filtered, pred_p1, pred_p2)
            
            st.metric("Expected Score", score_pred['expected_score'])
            
            st.write("**Top 5 Most Likely Scores:**")
            for score, prob in score_pred['top_predictions']:
                st.write(f"• {score}: {prob*100:.1f}%")
    
    # Upset Alert System
    st.subheader("🚨 Upset Alert System")
    upset_df = calculate_upset_potential(df_filtered, metrics['elo_ratings'])
    if not upset_df.empty and 'Upset Index' in upset_df.columns:
        # Format only columns that exist
        format_dict = {}
        if 'Favorite Win %' in upset_df.columns:
            format_dict['Favorite Win %'] = '{:.1f}%'
        if 'Upset Probability' in upset_df.columns:
            format_dict['Upset Probability'] = '{:.1f}%'
        if 'Underdog Form' in upset_df.columns:
            format_dict['Underdog Form'] = '{:.2f}'
        if 'Favorite Struggles' in upset_df.columns:
            format_dict['Favorite Struggles'] = '{:.2f}'
        if 'Upset Index' in upset_df.columns:
            format_dict['Upset Index'] = '{:.1f}'
        
        st.dataframe(
            upset_df.head(10).style.format(format_dict),
            use_container_width=True
        )
    else:
        st.info("🟢 No high upset potential detected. All matches look predictable!")
    
    # Performance Trajectory
    st.subheader("📈 Performance Trajectory Analysis")
    traj_player = st.selectbox("Select player for trajectory analysis", all_players, key="traj_player_analysis")
    trajectory = calculate_performance_trajectory(df_filtered, traj_player, 30)
    
    if 'error' not in trajectory:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Win Rate", f"{trajectory['current_win_rate']*100:.1f}%")
        with col2:
            st.metric("30-Day Projection", f"{trajectory['projected_win_rate']*100:.1f}%")
        with col3:
            st.metric("Trajectory", trajectory['trajectory'])
        
        st.write(f"**Confidence Interval:** {trajectory['confidence_interval'][0]*100:.1f}% - {trajectory['confidence_interval'][1]*100:.1f}%")


def display_performance_patterns(df_filtered: pd.DataFrame, metrics: Dict):
    """Display Performance Patterns analysis"""
    st.subheader("💪 Performance Patterns & Analytics")
    
    # Add info box explanation
    create_section_info(
        "Performance Patterns", 
        SECTION_EXPLANATIONS["Performance Patterns"]["description"],
        SECTION_EXPLANATIONS["Performance Patterns"]["metrics"]
    )
    
    pattern_tabs = st.tabs(["Clutch Performance", "Comebacks", "Fatigue Analysis", "Optimal Rest"])
    
    with pattern_tabs[0]:
        st.subheader("🎯 Clutch Performance Analysis")
        if not metrics['clutch'].empty:
            # Clutch performance chart
            clutch_chart = alt.Chart(metrics['clutch']).mark_bar().encode(
                x=alt.X('Player:N', sort='-y'),
                y='Clutch Rating:Q',
                color=alt.Color('Clutch Differential:Q', 
                               scale=alt.Scale(scheme='redblue', domain=[-0.3, 0.3])),
                tooltip=['Player:N', 'Clutch Win Rate:Q', 'Overall Win Rate:Q', 
                        'Clutch Differential:Q', 'Clutch Matches:Q']
            ).properties(
                title='Clutch Performance Ratings',
                height=400
            )
            st.altair_chart(clutch_chart, use_container_width=True)
            
            # Clutch stats table
            st.dataframe(
                metrics['clutch'].style.format({
                    'Clutch Win Rate': '{:.2%}',
                    'Overall Win Rate': '{:.2%}',
                    'Clutch Differential': '{:+.2%}',
                    'Clutch Rating': '{:.2f}'
                }),
                use_container_width=True
            )
    
    with pattern_tabs[1]:
        st.subheader("💪 Comeback Analysis")
        if not metrics['comebacks'].empty:
            # Comeback visualization
            comeback_chart = alt.Chart(metrics['comebacks']).mark_circle(size=200).encode(
                x='Comeback Rate:Q',
                y='Mental Toughness Score:Q',
                size='Comeback Opportunities:Q',
                color=alt.Color('Player:N'),
                tooltip=['Player:N', 'Comebacks:Q', 'Comeback Opportunities:Q', 
                        'Comeback Rate:Q', 'Biggest Comeback:Q']
            ).properties(
                title='Comeback Ability Analysis',
                height=400
            )
            st.altair_chart(comeback_chart, use_container_width=True)
    
    with pattern_tabs[2]:
        st.subheader("😮‍💨 Fatigue Analysis")
        if not metrics['fatigue'].empty:
            # Group by player and calculate average
            fatigue_summary = metrics['fatigue'].groupby('Player').agg({
                'Early Win Rate': 'mean',
                'Late Win Rate': 'mean',
                'Fatigue Impact': 'mean',
                'Endurance Score': 'mean'
            }).reset_index()
            
            # Fatigue impact chart
            fatigue_chart = alt.Chart(fatigue_summary).mark_bar().encode(
                x=alt.X('Player:N', sort='-y'),
                y='Fatigue Impact:Q',
                color=alt.condition(
                    alt.datum['Fatigue Impact'] > 0,
                    alt.value('#d62728'),  # Red for negative impact
                    alt.value('#2ca02c')   # Green for positive
                ),
                tooltip=['Player:N', 'Early Win Rate:Q', 'Late Win Rate:Q', 
                        'Fatigue Impact:Q', 'Endurance Score:Q']
            ).properties(
                title='Fatigue Impact on Performance',
                height=400
            )
            st.altair_chart(fatigue_chart, use_container_width=True)
    
    with pattern_tabs[3]:
        st.subheader("⏱️ Optimal Rest Period Analysis")
        if not metrics['optimal_rest'].empty:
            st.dataframe(
                metrics['optimal_rest'].style.format({
                    'Optimal Win Rate': '{:.2%}',
                    'Same Day Rate': '{:.2%}'
                }),
                use_container_width=True
            )


def display_psychological_insights(df_filtered: pd.DataFrame, metrics: Dict):
    """Display Psychological Insights section"""
    st.subheader("🧠 Psychological Insights & Mental Game")
    
    # Add info box explanation
    create_section_info(
        "Psychological Insights", 
        SECTION_EXPLANATIONS["Psychological Insights"]["description"],
        SECTION_EXPLANATIONS["Psychological Insights"]["metrics"]
    )
    
    psych_tabs = st.tabs(["Pressure Performance", "Nemesis Analysis", "Momentum Tracking"])
    
    with psych_tabs[0]:
        st.subheader("💎 Performance Under Pressure")
        if not metrics['pressure'].empty:
            # Pressure performance visualization
            pressure_chart = alt.Chart(metrics['pressure']).mark_bar().encode(
                x=alt.X('Player:N', sort='-y'),
                y='Mental Fortitude:Q',
                color=alt.Color('Pressure Win Rate:Q',
                               scale=alt.Scale(scheme='viridis')),
                tooltip=['Player:N', 'Pressure Win Rate:Q', 'Revenge Rate:Q',
                        'Mental Fortitude:Q']
            ).properties(
                title='Mental Fortitude Rankings',
                height=400
            )
            st.altair_chart(pressure_chart, use_container_width=True)
    
    with psych_tabs[1]:
        st.subheader("😈 Nemesis & Favorite Opponents")
        all_players = sorted(set(df_filtered["Player1"]) | set(df_filtered["Player2"]))
        selected_player_nemesis = st.selectbox("Select player for nemesis analysis", all_players, key="nemesis_player")
        
        if selected_player_nemesis in metrics['nemesis']:
            nemesis_df = metrics['nemesis'][selected_player_nemesis]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Toughest Opponents (Nemeses)**")
                st.dataframe(
                    nemesis_df.head(3)[['Opponent', 'Win Rate', 'Difficulty Score']].style.format({
                        'Win Rate': '{:.2%}',
                        'Difficulty Score': '{:.1f}'
                    })
                )
            
            with col2:
                st.write("**Favorite Opponents**")
                st.dataframe(
                    nemesis_df.tail(3)[['Opponent', 'Win Rate', 'Difficulty Score']].style.format({
                        'Win Rate': '{:.2%}',
                        'Difficulty Score': '{:.1f}'
                    })
                )
    
    with psych_tabs[2]:
        st.subheader("🔥 Momentum & Hot Streaks")
        if not metrics['momentum'].empty:
            # Sort by momentum score
            momentum_sorted = metrics['momentum'].sort_values('Momentum Score', ascending=False)
            
            # Momentum visualization
            for _, player in momentum_sorted.iterrows():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    st.write(f"**{player['Player']}**")
                with col2:
                    st.write(player['Last 10 Form'])
                with col3:
                    st.write(f"{player['Form %']:.0f}%")
                with col4:
                    indicator = create_momentum_indicator(player['Momentum Score'])
                    st.write(indicator)


def display_enhanced_player_analysis(df_filtered: pd.DataFrame, metrics: Dict = None):
    """Enhanced player analysis with multi-player comparison and H2H deep dive"""
    
    if metrics is None:
        metrics = calculate_all_advanced_metrics(df_filtered)
    
    all_players = sorted(set(df_filtered["Player1"]) | set(df_filtered["Player2"]))
    
    analysis_tabs = st.tabs(["Multi-Player Comparison", "Head-to-Head Deep Dive"])
    
    # Tab 1: Multi-Player Comparison
    with analysis_tabs[0]:
        st.subheader("Multi-Player Performance Comparison")
        
        # Add info box explanation
        create_section_info(
            "Multi-Player Comparison", 
            SECTION_EXPLANATIONS["Multi-Player Comparison"]["description"],
            SECTION_EXPLANATIONS["Multi-Player Comparison"]["metrics"]
        )
        
        comparison_players = st.multiselect(
            "Select players to compare (2-4 players)",
            all_players,
            default=all_players[:min(3, len(all_players))],
            max_selections=4,
            key="comparison_players_h2h"
        )
        
        if len(comparison_players) >= 2:
            # Radar chart comparison
            st.subheader("Multi-dimensional Performance Comparison")
            radar_fig = create_player_comparison_radar(df_filtered, comparison_players, metrics)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # Head-to-head matrix
            st.subheader("Head-to-Head Performance Matrix")
            h2h_heatmap = create_win_rate_opponent_heatmap(df_filtered)
            st.altair_chart(h2h_heatmap, use_container_width=True)
            
            # Detailed comparison table
            st.subheader("Detailed Metrics Comparison")
            comparison_data = []
            
            for player in comparison_players:
                player_data = {"Player": player}
                
                # Win rate
                player_matches = df_filtered[(df_filtered["Player1"] == player) | (df_filtered["Player2"] == player)]
                wins = len(player_matches[player_matches["Winner"] == player])
                player_data['Win Rate'] = f"{wins/len(player_matches)*100:.1f}%" if len(player_matches) > 0 else "0%"
                player_data['Matches'] = len(player_matches)
                
                # Add metrics from each dataframe
                if 'clutch' in metrics and not metrics['clutch'].empty:
                    player_clutch = metrics['clutch'][metrics['clutch']['Player'] == player]
                    if not player_clutch.empty:
                        player_data['Clutch Rating'] = f"{player_clutch.iloc[0]['Clutch Rating']:.2f}"
                
                if 'dominance' in metrics and not metrics['dominance'].empty:
                    player_dom = metrics['dominance'][metrics['dominance']['Player'] == player]
                    if not player_dom.empty:
                        player_data['Dominance'] = f"{player_dom.iloc[0]['Dominance Score']:.1f}"
                
                if 'consistency' in metrics and not metrics['consistency'].empty:
                    player_cons = metrics['consistency'][metrics['consistency']['Player'] == player]
                    if not player_cons.empty:
                        player_data['Consistency'] = f"{player_cons.iloc[0]['Consistency Rating']:.1f}%"
                
                if 'momentum' in metrics and not metrics['momentum'].empty:
                    player_mom = metrics['momentum'][metrics['momentum']['Player'] == player]
                    if not player_mom.empty:
                        player_data['Momentum'] = player_mom.iloc[0]['Hot Streak']
                
                if 'elo_ratings' in metrics and player in metrics['elo_ratings']:
                    player_data['Elo Rating'] = f"{metrics['elo_ratings'][player]:.0f}"
                
                comparison_data.append(player_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Please select at least 2 players to compare")
    
    # Tab 2: Head-to-Head Deep Dive
    with analysis_tabs[1]:
        st.subheader("Head-to-Head Deep Dive Analysis")
        
        # Add info box explanation
        create_section_info(
            "Head-to-Head Deep Dive", 
            SECTION_EXPLANATIONS["Head-to-Head Deep Dive"]["description"],
            SECTION_EXPLANATIONS["Head-to-Head Deep Dive"]["metrics"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            player1 = st.selectbox(
                "Select Player 1",
                [""] + all_players,
                format_func=lambda x: "Select..." if x == "" else x,
                key="h2h_player1"
            )
        with col2:
            player2 = st.selectbox(
                "Select Player 2",
                [""] + all_players,
                format_func=lambda x: "Select..." if x == "" else x,
                key="h2h_player2"
            )
        
        if player1 and player2 and player1 != player2:
            df_h2h = df_filtered[
                ((df_filtered["Player1"] == player1) & (df_filtered["Player2"] == player2)) |
                ((df_filtered["Player1"] == player2) & (df_filtered["Player2"] == player1))
            ]
            
            if df_h2h.empty:
                st.write(f"No head-to-head matches found between {player1} and {player2}.")
            else:
                # H2H Summary
                col1, col2, col3 = st.columns(3)
                
                p1_wins = len(df_h2h[df_h2h["Winner"] == player1])
                p2_wins = len(df_h2h[df_h2h["Winner"] == player2])
                total_h2h = len(df_h2h)
                
                with col1:
                    st.metric(f"{player1} Wins", p1_wins)
                    st.metric("Win Rate", f"{p1_wins/total_h2h*100:.1f}%")
                
                with col2:
                    st.metric("Total Matches", total_h2h)
                    st.metric("Last Match", df_h2h['date'].max().strftime('%Y-%m-%d'))
                
                with col3:
                    st.metric(f"{player2} Wins", p2_wins)
                    st.metric("Win Rate", f"{p2_wins/total_h2h*100:.1f}%")
                
                # Match Prediction
                st.subheader("Next Match Prediction")
                prob_data = calculate_head_to_head_probability(
                    df_filtered, player1, player2, metrics['elo_ratings']
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{player1} Win Probability", 
                             f"{prob_data['player1_probability']*100:.1f}%")
                with col2:
                    st.metric(f"{player2} Win Probability", 
                             f"{prob_data['player2_probability']*100:.1f}%")
                
                # Score Prediction
                score_pred = predict_score_distribution(df_filtered, player1, player2)
                st.metric("Most Likely Score", score_pred['expected_score'])
                
                # H2H Match History
                st.subheader("Match History")
                h2h_display = df_h2h[['date', 'Player1', 'Score1', 'Player2', 'Score2', 'Winner']].copy()
                h2h_display['date'] = h2h_display['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(h2h_display.sort_values('date', ascending=False), use_container_width=True)
                
                # H2H Specific Analysis
                generate_analysis_content(df_h2h, include_ratings=False)
        else:
            st.info("Please select two different players to compare their head-to-head statistics!")


def generate_analysis_content(df_filtered: pd.DataFrame, include_ratings: bool):
    """
    Creates the sub-tabs for Overall Overanalysis:
      1) Performance Overview (NEW)
      2) Ratings (Elo, Glicko2, TrueSkill) (if include_elo=True)
      3) Wins & Points
      4) Avg. Margin
      5) Win/Loss Streaks
      6) Endurance and Grit
      7) Records & Leaderboards
      8) Match Stats
      9) Predictive Analytics (NEW)
      10) Performance Patterns (NEW)
      11) Psychological Insights (NEW)
    """
    
    # Calculate metrics once
    metrics = calculate_all_advanced_metrics(df_filtered)
    
    list_of_tabs = [
        "Performance Overview",
        "Ratings",
        "Wins & Points",
        "Avg. Margin",
        "Win/Loss Streaks",
        "Endurance and Grit",
        "Records & Leaderboards",
        "Match Stats",
        "Predictive Analytics",
        "Performance Patterns",
        "Psychological Insights"
    ]
    if not include_ratings:
        # If we skip the rating systems entirely
        list_of_tabs.remove("Ratings")

    tabs = st.tabs(list_of_tabs)
    idx = 0

    # 1) PERFORMANCE OVERVIEW (NEW)
    with tabs[idx]:
        display_performance_overview(df_filtered, metrics)
    idx += 1

    # 2) RATINGS (Optional)
    if include_ratings:
        with tabs[idx]:
            display_ratings_tabs(df_filtered)
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

    # 8) MATCH STATS
    with tabs[idx]:
        display_match_stats(df_filtered)
    idx += 1
    
    # 9) PREDICTIVE ANALYTICS (NEW)
    with tabs[idx]:
        display_predictive_analytics(df_filtered, metrics)
    idx += 1
    
    # 10) PERFORMANCE PATTERNS (NEW)
    with tabs[idx]:
        display_performance_patterns(df_filtered, metrics)
    idx += 1
    
    # 11) PSYCHOLOGICAL INSIGHTS (NEW)
    with tabs[idx]:
        display_psychological_insights(df_filtered, metrics)
    idx += 1
