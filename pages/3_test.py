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

# Ensure date column is cast as datetime
df['date'] = pd.to_datetime(df['date'], format="%Y%m%d", errors='coerce')

# ----- BASIC CLEANUP -----
# Sometimes, multiple names or typos appear for the same person (like "Friede" vs "Friedemann").
# You can do some basic unify logic if needed:
# Example: unify "Friede" into "Friedemann"
df['Player1'] = df['Player1'].replace("Friede", "Friedemann")
df['Player2'] = df['Player2'].replace("Friede", "Friedemann")

# ----- ADD COLUMNS FOR WINNER/LOSER -----
def find_winner(row):
    return row['Player1'] if row['Score1'] > row['Score2'] else row['Player2']

def find_loser(row):
    return row['Player2'] if row['Score1'] > row['Score2'] else row['Player1']

df['Winner'] = df.apply(find_winner, axis=1)
df['Loser'] = df.apply(find_loser, axis=1)
df['WinnerScore'] = df[['Score1','Score2']].max(axis=1)
df['LoserScore'] = df[['Score1','Score2']].min(axis=1)
df['PointDiff'] = df['WinnerScore'] - df['LoserScore']

# ----- SIDEBAR OPTIONS -----
st.sidebar.header("Filters")
all_players = sorted(list(set(df['Player1']).union(set(df['Player2']))))
selected_players = st.sidebar.multiselect("Filter by Player(s)", all_players, default=all_players)

# Filter the DataFrame by selected players (only keep matches where at least one selected player took part)
df_filtered = df[(df['Player1'].isin(selected_players)) | (df['Player2'].isin(selected_players))]

# ----- KEY METRICS -----
st.subheader("Key Performance Metrics")
col1, col2, col3 = st.columns(3)

total_matches = len(df_filtered)
total_players_in_view = len(selected_players)
total_unique_players = df_filtered[['Player1','Player2']].stack().nunique()

col1.metric("Total Matches in View", total_matches)
col2.metric("Players Selected", total_players_in_view)
col3.metric("Unique Players in Filtered Data", total_unique_players)

# ----- WINS & POINTS PER PLAYER -----
st.subheader("Overall Wins & Total Points per Player")

# Total Wins
wins_df = df_filtered.groupby('Winner').size().reset_index(name='Wins')

# Total Points Scored
# We'll sum up points from either Score1 or Score2. A quick way is to group
# by Player1 summing Score1 plus group by Player2 summing Score2, then merge.
points_p1 = df_filtered.groupby('Player1')['Score1'].sum().reset_index()
points_p1.columns = ['Player', 'Points']
points_p2 = df_filtered.groupby('Player2')['Score2'].sum().reset_index()
points_p2.columns = ['Player', 'Points']
total_points = pd.concat([points_p1, points_p2], ignore_index=True)
total_points = total_points.groupby('Player')['Points'].sum().reset_index()

# Merge the two metrics
summary_df = pd.merge(wins_df, total_points, left_on='Winner', right_on='Player', how='outer').drop(columns='Player')
summary_df.rename(columns={'Winner': 'Player'}, inplace=True)

# Some players may have never won but still appear in total_points, so let's combine carefully
# We'll do a full outer merge of (Wins table) and (Points table) to ensure nobody is lost:
# We already did it, but let's finalize it properly:
# If someone doesn't appear in 'Winner', then 'Wins' is NaN => fill with 0
summary_df['Wins'] = summary_df['Wins'].fillna(0).astype(int)

# Now check if there are players in total_points who never appear as winners. 
# Actually, the code above should handle that. If we still need them, we handle them. 
# But let's see if the final summary covers all players. 
# We do an outer join with total_points again to ensure coverage:
final_summary = pd.merge(
    total_points, 
    summary_df[['Player','Wins']], 
    on='Player', how='outer'
)
final_summary['Wins'] = final_summary['Wins'].fillna(0).astype(int)

# Sort by Wins desc
final_summary.sort_values(by='Wins', ascending=False, inplace=True)
final_summary.reset_index(drop=True, inplace=True)

# Display in Streamlit
st.dataframe(final_summary.style.format({"Points": "{:.0f}", "Wins": "{:.0f}"}), use_container_width=True)

# ----- HEAD-TO-HEAD MATCHUPS -----
st.subheader("Head-to-Head Matchups")
st.markdown("See how players perform against each other (win count).")

# We can create a pivot table counting how many times a player has beaten another
h2h_df = df_filtered.groupby(['Winner','Loser']).size().reset_index(name='Wins_against')
pivot_h2h = h2h_df.pivot(index='Winner', columns='Loser', values='Wins_against').fillna(0)

st.dataframe(pivot_h2h.astype(int), use_container_width=True)

# ----- DOMINATION METRICS -----
st.subheader("Domination Metrics")
st.markdown("**Average Margin of Victory** (among the selected filters).")

domination = df_filtered.groupby('Winner')['PointDiff'].mean().reset_index()
domination.columns = ['Player','Avg_Point_Diff']
domination.sort_values(by='Avg_Point_Diff', ascending=False, inplace=True)

st.dataframe(domination.style.format({"Avg_Point_Diff": "{:.2f}"}), use_container_width=True)

# ----- TIME-SERIES VIEW -----
st.subheader("Matches Over Time")
st.markdown("Number of matches played on each date (filtered by player selection).")

matches_over_time = df_filtered.groupby('date').size().reset_index(name='Matches')
c = alt.Chart(matches_over_time).mark_bar().encode(
    x='date:T',
    y='Matches:Q',
    tooltip=['date:T','Matches:Q']
).properties(
    width='container',
    height=300
)
st.altair_chart(c, use_container_width=True)

st.markdown("---")
st.markdown("""
**Ideas to extend**:
- Show longest winning streak per player
- Show best comeback (lowest to highest final difference)
- Show distribution of scores (histogram of all match point totals)
- More advanced visualizations!
""")
