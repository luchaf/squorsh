import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
from itertools import combinations

from streamlit_gsheets import GSheetsConnection

# Create GSheets connection
conn = st.connection("gsheets", type=GSheetsConnection)
worksheet_name = "match_results"
df = conn.read(worksheet=worksheet_name)

# ---- Convert date column to datetime ----
df['date'] = pd.to_datetime(df['date'], format="%Y%m%d", errors='coerce')

# ---- Basic data cleanup / unify names if needed ----
df['Player1'] = df['Player1'].replace("Friede", "Friedemann")
df['Player2'] = df['Player2'].replace("Friede", "Friedemann")

# ---- Derive winner / loser columns, margin, etc. ----
df['Winner'] = df.apply(lambda row: row['Player1'] if row['Score1'] > row['Score2'] else row['Player2'], axis=1)
df['Loser'] = df.apply(lambda row: row['Player2'] if row['Score1'] > row['Score2'] else row['Player1'], axis=1)
df['WinnerScore'] = df[['Score1', 'Score2']].max(axis=1)
df['LoserScore'] = df[['Score1', 'Score2']].min(axis=1)
df['PointDiff'] = df['WinnerScore'] - df['LoserScore']
df['LoserPointDiff'] = df['LoserScore'] - df['WinnerScore']

# ---- Compute the Day of the Week for each match date ----
df['day_of_week'] = df['date'].dt.day_name()

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Filters")

# 1) Filter on days of the week
all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
days_present_in_data = sorted(set(df['day_of_week']) & set(all_days))
selected_days = st.sidebar.multiselect(
    "Select Day(s) of the Week to Include",
    options=days_present_in_data,
    default=days_present_in_data
)

# Filter the DataFrame by the chosen days
df_day_filtered = df[df['day_of_week'].isin(selected_days)]

# 2) Filter on players
all_players = sorted(set(df_day_filtered['Player1']) | set(df_day_filtered['Player2']))
selected_players = st.sidebar.multiselect(
    "Select Player(s) to Include", 
    options=all_players, 
    default=all_players
)

# Filter only matches where at least one selected player participated
df_filtered = df_day_filtered[
    (df_day_filtered['Player1'].isin(selected_players)) |
    (df_day_filtered['Player2'].isin(selected_players))
]

# ---- TABS for organization ----
tab_main, tab_extensions = st.tabs(["Main Metrics", "Extended Stats"])

# =========================
#       TAB: MAIN
# =========================
with tab_main:
    st.subheader("Key Performance Metrics")

    # Quick info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches (Filtered)", len(df_filtered))
    col2.metric("Players Selected", len(selected_players))
    col3.metric(
        "Unique Players (Filtered)", 
        df_filtered[['Player1', 'Player2']].stack().nunique()
    )

    # ----- Wins & Points per Player -----
    st.subheader("Overall Wins & Total Points per Player")

    # Calculate total wins
    wins_df = df_filtered.groupby('Winner').size().reset_index(name='Wins')

    # Calculate total points
    points_p1 = df_filtered.groupby('Player1')['Score1'].sum().reset_index()
    points_p1.columns = ['Player', 'Points']
    points_p2 = df_filtered.groupby('Player2')['Score2'].sum().reset_index()
    points_p2.columns = ['Player', 'Points']
    total_points = pd.concat([points_p1, points_p2], ignore_index=True)
    total_points = total_points.groupby('Player')['Points'].sum().reset_index()

    # Merge Wins + Points
    summary_df = pd.merge(
        wins_df, 
        total_points, 
        left_on='Winner', 
        right_on='Player', 
        how='outer'
    ).drop(columns='Player')
    summary_df.rename(columns={'Winner': 'Player'}, inplace=True)
    summary_df['Wins'] = summary_df['Wins'].fillna(0).astype(int)

    # Some players might never have won, ensure we capture them
    final_summary = pd.merge(
        total_points, 
        summary_df[['Player', 'Wins']], 
        on='Player', how='outer'
    )
    final_summary['Wins'] = final_summary['Wins'].fillna(0).astype(int)

    # Sort by wins desc
    final_summary.sort_values('Wins', ascending=False, inplace=True, ignore_index=True)

    st.dataframe(
        final_summary.style.format({"Points": "{:.0f}", "Wins": "{:.0f}"}), 
        use_container_width=True
    )

    # Enhanced graph for Wins & Points per Player
    chart_points_wins = alt.Chart(final_summary).mark_bar().encode(
        x=alt.X('Points:Q', title='Total Points'),
        y=alt.Y('Player:N', sort='-x', title='Player'),
        color='Wins:Q',
        tooltip=['Player', 'Points', 'Wins']
    ).properties(width='container', height=400)
    st.altair_chart(chart_points_wins, use_container_width=True)

    # ----- Head-to-Head Wins -----
    st.subheader("Head-to-Head Wins")

    h2h_df = df_filtered.groupby(['Winner', 'Loser']).size().reset_index(name='Wins_against')
    pivot_h2h = h2h_df.pivot(index='Winner', columns='Loser', values='Wins_against').fillna(0).astype(int)
    st.dataframe(pivot_h2h, use_container_width=True)

    # Enhanced heatmap for Head-to-Head Wins
    heatmap_h2h = alt.Chart(h2h_df).mark_rect().encode(
        x=alt.X('Loser:N', title='Loser'),
        y=alt.Y('Winner:N', title='Winner'),
        color=alt.Color('Wins_against:Q', title='Wins', scale=alt.Scale(scheme='blues')),
        tooltip=['Winner', 'Loser', 'Wins_against']
    ).properties(width='container', height=400)
    st.altair_chart(heatmap_h2h, use_container_width=True)

    # ----- Matches Over Time -----
    st.subheader("Matches Over Time (Filtered)")
    matches_over_time = df_filtered.groupby('date').size().reset_index(name='Matches')
    chart_with_trendline = alt.Chart(matches_over_time).mark_line(point=True).encode(
        x='date:T',
        y='Matches:Q',
        tooltip=['date:T', 'Matches:Q']
    ).properties(width='container', height=300)
    st.altair_chart(chart_with_trendline, use_container_width=True)

# =========================
#    TAB: EXTENDED STATS
# =========================
with tab_extensions:
    st.header("Extended Stats & Fun Analyses")

    # Longest streaks
    st.subheader("Longest Winning/Losing Streaks per Player")

    def compute_longest_streaks(df_input, player, streak_type):
        df_player = df_input[(df_input['Player1'] == player) | (df_input['Player2'] == player)]
        df_player = df_player.sort_values('date', ascending=True)
        longest = 0
        current = 0

        for _, row in df_player.iterrows():
            if (streak_type == 'win' and row['Winner'] == player) or (streak_type == 'loss' and row['Loser'] == player):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    streaks = []
    for player in all_players:
        w_streak = compute_longest_streaks(df_filtered, player, 'win')
        l_streak = compute_longest_streaks(df_filtered, player, 'loss')
        streaks.append((player, w_streak, l_streak))

    df_streaks = pd.DataFrame(streaks, columns=['Player', 'Longest_Win_Streak', 'Longest_Loss_Streak'])
    df_streaks.sort_values('Longest_Win_Streak', ascending=False, inplace=True, ignore_index=True)

    st.dataframe(df_streaks, use_container_width=True)

    # Enhanced bar chart for streaks
    streak_chart = alt.Chart(df_streaks).transform_fold(
        ['Longest_Win_Streak', 'Longest_Loss_Streak'],
        as_=['Streak Type', 'Streak Length']
    ).mark_bar().encode(
        x=alt.X('Streak Length:Q', title='Length'),
        y=alt.Y('Player:N', sort='-x', title='Player'),
        color='Streak Type:N',
        tooltip=['Player', 'Streak Type', 'Streak Length']
    ).properties(width='container', height=400)
    st.altair_chart(streak_chart, use_container_width=True)
