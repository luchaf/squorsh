import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from pointless_utils import (
    extract_data_from_games,
    get_name_opponent_name_df,
    get_name_streaks_df,
    calculate_combination_stats,
    derive_results,
    win_loss_trends_plot,
    wins_and_losses_over_time_plot,
    graph_win_and_loss_streaks,
    plot_player_combo_graph,
    plot_bars,
    cumulative_wins_over_time,
    cumulative_win_ratio_over_time,
    entities_face_to_face_over_time_abs,
    entities_face_to_face_over_time_rel,
    closeness_of_matches_over_time,
    correct_names_in_dataframe,
)
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

# Create GSheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

worksheet_name = "match_results"
df = conn.read(worksheet=worksheet_name)

# 1) Ensure date column is parsed as datetime
df['date'] = pd.to_datetime(df['date'], format="%Y%m%d", errors='coerce')

# 2) Basic cleanup or name-unification if needed:
# (Example: unify "Friede" -> "Friedemann", in case your data has that mismatch)
df['Player1'] = df['Player1'].replace("Friede", "Friedemann")
df['Player2'] = df['Player2'].replace("Friede", "Friedemann")

# 3) Determine Winner, Loser, and additional columns
df['Winner'] = df.apply(lambda row: row['Player1'] if row['Score1'] > row['Score2'] else row['Player2'], axis=1)
df['Loser'] = df.apply(lambda row: row['Player2'] if row['Score1'] > row['Score2'] else row['Player1'], axis=1)
df['WinnerScore'] = df[['Score1','Score2']].max(axis=1)
df['LoserScore'] = df[['Score1','Score2']].min(axis=1)
df['PointDiff'] = df['WinnerScore'] - df['LoserScore']

# 4) Sidebar filter
st.sidebar.header("Filters")
all_players = sorted(set(df['Player1']) | set(df['Player2']))
selected_players = st.sidebar.multiselect("Select Player(s) to Include", all_players, default=all_players)

# Filter only matches where at least one selected player participated
df_filtered = df[
    (df['Player1'].isin(selected_players)) |
    (df['Player2'].isin(selected_players))
]

# ----- KEY METRICS -----
st.subheader("Key Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Matches in View", len(df_filtered))
col2.metric("Players Selected", len(selected_players))
col3.metric("Unique Players (Filtered)", df_filtered[['Player1','Player2']].stack().nunique())

# ----- WINS & TOTAL POINTS PER PLAYER -----
st.subheader("Overall Wins & Total Points per Player")

# Calculate total wins
wins_df = df_filtered.groupby('Winner').size().reset_index(name='Wins')

# Calculate total points scored
points_p1 = df_filtered.groupby('Player1')['Score1'].sum().reset_index()
points_p1.columns = ['Player', 'Points']
points_p2 = df_filtered.groupby('Player2')['Score2'].sum().reset_index()
points_p2.columns = ['Player', 'Points']
total_points = pd.concat([points_p1, points_p2], ignore_index=True)
total_points = total_points.groupby('Player')['Points'].sum().reset_index()

# Merge the two metrics into one summary
summary_df = pd.merge(wins_df, total_points, left_on='Winner', right_on='Player', how='outer').drop(columns='Player')
summary_df.rename(columns={'Winner': 'Player'}, inplace=True)
summary_df['Wins'] = summary_df['Wins'].fillna(0).astype(int)

# Make sure we include players who never won but appear in total_points
final_summary = pd.merge(total_points, summary_df[['Player','Wins']], on='Player', how='outer')
final_summary['Wins'] = final_summary['Wins'].fillna(0).astype(int)

# Sort by number of wins descending
final_summary.sort_values(by='Wins', ascending=False, inplace=True)
final_summary.reset_index(drop=True, inplace=True)

st.dataframe(final_summary.style.format({"Points":"{:.0f}","Wins":"{:.0f}"}), use_container_width=True)

# ----- HEAD-TO-HEAD: WINS COUNT -----
st.subheader("Head-to-Head Wins")
st.markdown("Shows how many times a player (row) has **defeated** another player (column).")

h2h_df = df_filtered.groupby(['Winner','Loser']).size().reset_index(name='Wins_against')
pivot_h2h = h2h_df.pivot(index='Winner', columns='Loser', values='Wins_against').fillna(0).astype(int)
st.dataframe(pivot_h2h, use_container_width=True)

# ----- AVERAGE MARGIN OF VICTORY & DEFEAT PER PLAYER -----
st.subheader("Average Margin of Victory & Defeat (Per Player)")

# Add a column from the loser's perspective
df_filtered['LoserPointDiff'] = df_filtered['LoserScore'] - df_filtered['WinnerScore']

# Average margin of victory
df_margin_vic = df_filtered.groupby('Winner')['PointDiff'].mean().reset_index()
df_margin_vic.columns = ['Player', 'Avg_margin_victory']

# Average margin of defeat
df_margin_def = df_filtered.groupby('Loser')['LoserPointDiff'].mean().reset_index()
df_margin_def.columns = ['Player', 'Avg_margin_defeat']

# Merge the two sets
df_margin_summary = pd.merge(df_margin_vic, df_margin_def, on='Player', how='outer')
df_margin_summary[['Avg_margin_victory','Avg_margin_defeat']] = df_margin_summary[
    ['Avg_margin_victory','Avg_margin_defeat']
].fillna(0)
df_margin_summary.sort_values(by='Player', inplace=True)

st.dataframe(
    df_margin_summary.style.format({"Avg_margin_victory":"{:.2f}", "Avg_margin_defeat":"{:.2f}"}),
    use_container_width=True
)

# ----- HEAD-TO-HEAD: AVERAGE MARGINS -----
st.subheader("Head-to-Head: Average Margin of Victory")

df_matchup_margin = (
    df_filtered
    .groupby(['Winner','Loser'])['PointDiff']
    .mean()
    .reset_index(name='Avg_margin_of_victory')
)

matchup_pivot_vic = df_matchup_margin.pivot(index='Winner', columns='Loser', values='Avg_margin_of_victory').fillna(0)
matchup_pivot_vic = matchup_pivot_vic.round(2)
st.dataframe(matchup_pivot_vic, use_container_width=True)

st.subheader("Head-to-Head: Average Margin of Defeat")

df_matchup_loss = (
    df_filtered
    .groupby(['Loser','Winner'])['LoserPointDiff']
    .mean()
    .reset_index(name='Avg_margin_of_defeat')
)

matchup_pivot_def = df_matchup_loss.pivot(index='Loser', columns='Winner', values='Avg_margin_of_defeat').fillna(0)
matchup_pivot_def = matchup_pivot_def.round(2)
st.dataframe(matchup_pivot_def, use_container_width=True)

# ----- MATCHES OVER TIME -----
st.subheader("Matches Over Time")

# Count how many matches happened on each date in the filtered set
matches_over_time = df_filtered.groupby('date').size().reset_index(name='Matches')
chart = alt.Chart(matches_over_time).mark_bar().encode(
    x='date:T',
    y='Matches:Q',
    tooltip=['date:T','Matches:Q']
).properties(
    width='container',
    height=300
)
st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.markdown("""
**Possible Extensions**  
- Display winning streaks per player.  
- Show distribution of total points scored in matches.  
- Identify the 'closest' matches (smallest margin).  
- Calculate more advanced stats like Elo ratings, etc.  
""")

