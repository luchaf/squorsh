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
df['date'] = pd.to_datetime(df['date'], format="%Y%m%d", errors='coerce')

# ---- Basic data cleanup ----
df['Player1'] = df['Player1'].replace("Friede", "Friedemann")
df['Player2'] = df['Player2'].replace("Friede", "Friedemann")

# ---- Derive winner/loser columns ----
df['Winner'] = df.apply(lambda row: row['Player1'] if row['Score1'] > row['Score2'] else row['Player2'], axis=1)
df['Loser'] = df.apply(lambda row: row['Player2'] if row['Score1'] > row['Score2'] else row['Player1'], axis=1)
df['WinnerScore'] = df[['Score1', 'Score2']].max(axis=1)
df['LoserScore'] = df[['Score1', 'Score2']].min(axis=1)
df['PointDiff'] = df['WinnerScore'] - df['LoserScore']
df['LoserPointDiff'] = df['LoserScore'] - df['WinnerScore']
df['day_of_week'] = df['date'].dt.day_name()

# ---- FILTERS (Top Section) ----
st.sidebar.header("Filters")

# Date & Time Filters
days_present_in_data = sorted(set(df['day_of_week']))
selected_days = st.sidebar.multiselect(
    "Select Day(s) of the Week to Include",
    options=days_present_in_data,
    default=days_present_in_data
)

# Player Filters
all_players = sorted(set(df['Player1']) | set(df['Player2']))
selected_players = st.sidebar.multiselect(
    "Select Player(s) to Include", 
    options=all_players, 
    default=all_players
)

# Apply Filters
df_filtered = df[
    (df['day_of_week'].isin(selected_days)) &
    ((df['Player1'].isin(selected_players)) | (df['Player2'].isin(selected_players)))
]

# ---- ORGANIZATION: TABS ----
tab_summary, tab_head_to_head, tab_player_perf, tab_match_stats, tab_advanced = st.tabs([
    "Summary Metrics", "Head-to-Head", "Player Performance", "Match Statistics", "Advanced Analytics"
])

# =========================
#       TAB: SUMMARY
# =========================
with tab_summary:
    st.subheader("Summary Metrics")

    # Quick Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches (Filtered)", len(df_filtered))
    col2.metric("Players Selected", len(selected_players))
    col3.metric(
        "Unique Players (Filtered)", 
        df_filtered[['Player1', 'Player2']].stack().nunique()
    )

    # Wins & Points Summary
    st.subheader("Wins & Points Per Player")
    wins_df = df_filtered.groupby('Winner').size().reset_index(name='Wins')
    points_p1 = df_filtered.groupby('Player1')['Score1'].sum().reset_index()
    points_p1.columns = ['Player', 'Points']
    points_p2 = df_filtered.groupby('Player2')['Score2'].sum().reset_index()
    points_p2.columns = ['Player', 'Points']
    total_points = pd.concat([points_p1, points_p2], ignore_index=True).groupby('Player')['Points'].sum().reset_index()

    summary_df = pd.merge(wins_df, total_points, left_on='Winner', right_on='Player', how='outer').drop(columns='Player')
    summary_df.rename(columns={'Winner': 'Player'}, inplace=True)
    summary_df['Wins'] = summary_df['Wins'].fillna(0).astype(int)
    final_summary = pd.merge(total_points, summary_df[['Player', 'Wins']], on='Player', how='outer')
    final_summary['Wins'] = final_summary['Wins'].fillna(0).astype(int)
    final_summary.sort_values('Wins', ascending=False, inplace=True, ignore_index=True)

    st.dataframe(
        final_summary.style.format({"Points": "{:.0f}", "Wins": "{:.0f}"}),
        use_container_width=True
    )


    
    # Sort players by number of wins
    final_summary.sort_values(by='Wins', ascending=False, inplace=True)
    
    # Melt the dataframe to a long format for Altair
    chart_data = final_summary.melt(
        id_vars='Player', 
        value_vars=['Wins', 'Points'], 
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create the bar chart using Altair
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Player:N', sort=list(final_summary['Player']), title='Player'),
        y=alt.Y('Value:Q', title='Count'),
        color='Metric:N',
        column=alt.Column('Metric:N', title='Metric', header=alt.Header(labelOrient="bottom"))
    ).properties(
        width=alt.Step(40)  # Adjust bar width
    )
    
    st.altair_chart(bar_chart, use_container_width=True)


# =========================
#    TAB: HEAD-TO-HEAD
# =========================
with tab_head_to_head:
    st.subheader("Head-to-Head Wins")
    h2h_df = df_filtered.groupby(['Winner', 'Loser']).size().reset_index(name='Wins_against')
    pivot_h2h = h2h_df.pivot(index='Winner', columns='Loser', values='Wins_against').fillna(0).astype(int)
    st.dataframe(pivot_h2h, use_container_width=True)

    st.subheader("Head-to-Head Margins")
    df_matchup_margin = (
        df_filtered
        .groupby(['Winner', 'Loser'])['PointDiff']
        .mean()
        .reset_index(name='Avg_margin_of_victory')
    )
    pivot_margin_vic = df_matchup_margin.pivot(index='Winner', columns='Loser', values='Avg_margin_of_victory').fillna(0).round(2)
    st.dataframe(pivot_margin_vic, use_container_width=True)

# =========================
#   TAB: PLAYER PERFORMANCE
# =========================
with tab_player_perf:
    st.subheader("Winning and Losing Streaks")

    df_sorted = df_filtered.sort_values(['date'], ascending=True)
    streaks = []
    unique_players = sorted(set(df_filtered['Player1']) | set(df_filtered['Player2']))

    for player in unique_players:
        current_win, max_win = 0, 0
        current_loss, max_loss = 0, 0

        for _, row in df_sorted.iterrows():
            if row['Winner'] == player:
                current_win += 1
                max_win = max(max_win, current_win)
                current_loss = 0
            elif row['Loser'] == player:
                current_loss += 1
                max_loss = max(max_loss, current_loss)
                current_win = 0

        streaks.append((player, max_win, max_loss))

    streaks_df = pd.DataFrame(streaks, columns=['Player', 'Longest_Win_Streak', 'Longest_Loss_Streak'])
    streaks_df.sort_values('Longest_Win_Streak', ascending=False, inplace=True)
    st.dataframe(streaks_df, use_container_width=True)

    
    st.subheader("Performance by Nth Match of Day")

    def meltdown_day_matches(df_in):
        df_in = df_in.sort_values(['date','match_number_total','match_number_day'], ascending=True)

        df_p1 = df_in[['date','Player1','Winner','Loser','Score1','Score2','match_number_total','match_number_day']]
        df_p1 = df_p1.rename(columns={
            'Player1':'player',
            'Score1':'score_for_this_player',
            'Score2':'score_for_opponent'
        })
        df_p1['did_win'] = (df_p1['player'] == df_p1['Winner']).astype(int)

        df_p2 = df_in[['date','Player2','Winner','Loser','Score1','Score2','match_number_total','match_number_day']]
        df_p2 = df_p2.rename(columns={
            'Player2':'player',
            'Score2':'score_for_this_player',
            'Score1':'score_for_opponent'
        })
        df_p2['did_win'] = (df_p2['player'] == df_p2['Winner']).astype(int)

        df_stacked = pd.concat([df_p1, df_p2], ignore_index=True)

        df_stacked = df_stacked.sort_values(['date','player','match_number_total','match_number_day'])
        df_stacked['MatchOfDay'] = df_stacked.groupby(['date','player']).cumcount() + 1

        return df_stacked

    df_daycount = meltdown_day_matches(df_filtered)

    df_day_agg = (
        df_daycount
        .groupby(['player','MatchOfDay'])['did_win']
        .agg(['sum','count'])
        .reset_index()
    )
    df_day_agg['win_rate'] = df_day_agg['sum'] / df_day_agg['count']

    # Let user select which players to show in the chart
    available_players = sorted(df_day_agg['player'].unique())
    players_for_nth_chart = st.multiselect(
        "Select which players to display in the Nth-Match-of-Day chart",
        options=available_players,
        default=available_players
    )

    if players_for_nth_chart:
        df_day_agg_display = df_day_agg[df_day_agg['player'].isin(players_for_nth_chart)]

        base = alt.Chart(df_day_agg_display).encode(
            x=alt.X('MatchOfDay:Q', title='Nth Match of the Day'),
            y=alt.Y('win_rate:Q', title='Win Rate (0-1)'),
            color=alt.Color('player:N', title='Player'),
            tooltip=[
                alt.Tooltip('player:N'),
                alt.Tooltip('MatchOfDay:Q'),
                alt.Tooltip('win_rate:Q', format='.2f'),
                alt.Tooltip('sum:Q', title='Wins'),
                alt.Tooltip('count:Q', title='Matches')
            ]
        )

        # 1) Actual data line with points
        lines_layer = base.mark_line(point=True)

        # 2) Regression line
        trend_layer = (
            base
            .transform_regression(
                'MatchOfDay', 'win_rate', groupby=['player']
            )
            .mark_line(strokeDash=[4,4])
            .encode(opacity=alt.value(0.7))
        )

        chart_match_of_day = alt.layer(lines_layer, trend_layer).properties(
            width='container',
            height=400
        )

        st.altair_chart(chart_match_of_day, use_container_width=True)

        st.markdown("**Table**: Win Rate by Nth Match of Day (Filtered)")
        st.dataframe(
            df_day_agg_display[['player','MatchOfDay','sum','count','win_rate']]
            .sort_values(['player','MatchOfDay'])
            .reset_index(drop=True)
            .style.format({'win_rate':'{:.2f}'}),
            use_container_width=True
        )
    else:
        st.info("No players selected for the Nth-match-of-day chart.")

    st.markdown("""
    This chart & table show how each **selected** player performs in their 1st, 2nd, 3rd, etc. match **per day**.  
    The **solid line** is their actual data, and the **dashed line** is a linear trend line.  
    """)

# =========================
#   TAB: MATCH STATISTICS
# =========================
with tab_match_stats:
    st.subheader("Match Result Distribution")
    df_filtered['ResultPair'] = df_filtered.apply(lambda row: f"{int(max(row['Score1'], row['Score2']))}:{int(min(row['Score1'], row['Score2']))}", axis=1)
    pair_counts = df_filtered['ResultPair'].value_counts().reset_index()
    pair_counts.columns = ['ResultPair', 'Count']

    results_chart = alt.Chart(pair_counts).mark_bar().encode(
        x=alt.X('Count:Q', title='Number of Matches'),
        y=alt.Y('ResultPair:N', sort='-x', title='Score Category'),
        tooltip=['ResultPair', 'Count']
    )
    st.altair_chart(results_chart, use_container_width=True)

    # === 3) Identify the 'closest' matches (smallest margin) ===
    st.subheader("Closest Matches (Filtered)")

    n_closest = st.slider("Number of closest matches to display", min_value=1, max_value=50, value=20)
    df_filtered['TotalPoints'] = df_filtered['Score1'] + df_filtered['Score2']

    # Sort by margin ascending, then by total points descending
    df_closest_sorted = df_filtered.sort_values(['PointDiff','TotalPoints'], ascending=[True, False])
    closest_subset = df_closest_sorted.head(n_closest)

    st.dataframe(
        closest_subset[[
            'match_number_total','date','Player1','Score1','Player2','Score2','PointDiff','TotalPoints'
        ]].reset_index(drop=True),
        use_container_width=True
    )



# =========================
#  TAB: ADVANCED ANALYTICS
# =========================
with tab_advanced:
    st.subheader("Elo Ratings")
    df_sorted = df.sort_values(['date'], ascending=True)
    elo_ratings = defaultdict(lambda: 1500)
    K = 20

    for _, row in df_sorted.iterrows():
        p1, p2 = row['Player1'], row['Player2']
        r1, r2 = elo_ratings[p1], elo_ratings[p2]
        exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        exp2 = 1 / (1 + 10 ** ((r1 - r2) / 400))

        if row['Winner'] == p1:
            elo_ratings[p1] += K * (1 - exp1)
            elo_ratings[p2] += K * (0 - exp2)
        else:
            elo_ratings[p1] += K * (0 - exp1)
            elo_ratings[p2] += K * (1 - exp2)

    elo_df = pd.DataFrame([(player, rating) for player, rating in elo_ratings.items()], columns=['Player', 'Elo_Rating'])
    elo_df.sort_values('Elo_Rating', ascending=False, inplace=True)

    st.dataframe(elo_df, use_container_width=True)
