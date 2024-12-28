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

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict

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
df['WinnerScore'] = df[['Score1','Score2']].max(axis=1)
df['LoserScore'] = df[['Score1','Score2']].min(axis=1)
df['PointDiff'] = df['WinnerScore'] - df['LoserScore']
df['LoserPointDiff'] = df['LoserScore'] - df['WinnerScore']  # typically negative

# ---- Sidebar Filters ----
st.sidebar.header("Filters")
all_players = sorted(set(df['Player1']) | set(df['Player2']))
selected_players = st.sidebar.multiselect(
    "Select Player(s) to Include", 
    options=all_players, 
    default=all_players
)

# Filter only matches where at least one selected player participated
df_filtered = df[
    (df['Player1'].isin(selected_players)) |
    (df['Player2'].isin(selected_players))
]

# ---- TABS for organization ----
tab_main, tab_extensions = st.tabs(["Main Metrics", "Extended Stats"])

# =========================
#       TAB: MAIN
# =========================
with tab_main:
    st.subheader("Key Performance Metrics")

    # Some quick info
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches (Filtered)", len(df_filtered))
    col2.metric("Players Selected", len(selected_players))
    col3.metric(
        "Unique Players (Filtered)", 
        df_filtered[['Player1','Player2']].stack().nunique()
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
        summary_df[['Player','Wins']], 
        on='Player', how='outer'
    )
    final_summary['Wins'] = final_summary['Wins'].fillna(0).astype(int)

    # Sort by wins desc
    final_summary.sort_values('Wins', ascending=False, inplace=True, ignore_index=True)

    st.dataframe(
        final_summary.style.format({"Points":"{:.0f}","Wins":"{:.0f}"}), 
        use_container_width=True
    )

    # ----- Head-to-Head Wins -----
    st.subheader("Head-to-Head Wins")
    st.markdown("**Rows:** Winner, **Columns:** Loser → number of times row player beat column player.")
    h2h_df = df_filtered.groupby(['Winner','Loser']).size().reset_index(name='Wins_against')
    pivot_h2h = h2h_df.pivot(index='Winner', columns='Loser', values='Wins_against').fillna(0).astype(int)
    st.dataframe(pivot_h2h, use_container_width=True)

    # ----- Avg Margin of Victory & Defeat (Per Player) -----
    st.subheader("Average Margin of Victory & Defeat (Per Player)")

    df_margin_vic = df_filtered.groupby('Winner')['PointDiff'].mean().reset_index()
    df_margin_vic.columns = ['Player','Avg_margin_victory']

    df_margin_def = df_filtered.groupby('Loser')['LoserPointDiff'].mean().reset_index()
    df_margin_def.columns = ['Player','Avg_margin_defeat']

    df_margin_summary = pd.merge(df_margin_vic, df_margin_def, on='Player', how='outer').fillna(0)
    df_margin_summary.sort_values('Player', inplace=True)

    st.dataframe(
        df_margin_summary.style.format({"Avg_margin_victory":"{:.2f}", "Avg_margin_defeat":"{:.2f}"}),
        use_container_width=True
    )

    # ----- Head-to-Head: Average Margin (Winner & Loser perspective) -----
    st.subheader("Head-to-Head: Average Margin of Victory")

    df_matchup_margin = (
        df_filtered
        .groupby(['Winner','Loser'])['PointDiff']
        .mean()
        .reset_index(name='Avg_margin_of_victory')
    )
    pivot_margin_vic = df_matchup_margin.pivot(index='Winner', columns='Loser', values='Avg_margin_of_victory').fillna(0).round(2)
    st.dataframe(pivot_margin_vic, use_container_width=True)

    st.subheader("Head-to-Head: Average Margin of Defeat")

    df_matchup_loss = (
        df_filtered
        .groupby(['Loser','Winner'])['LoserPointDiff']
        .mean()
        .reset_index(name='Avg_margin_of_defeat')
    )
    pivot_margin_def = df_matchup_loss.pivot(index='Loser', columns='Winner', values='Avg_margin_of_defeat').fillna(0).round(2)
    st.dataframe(pivot_margin_def, use_container_width=True)

    # ----- Matches Over Time -----
    st.subheader("Matches Over Time (Filtered)")
    matches_over_time = df_filtered.groupby('date').size().reset_index(name='Matches')
    chart = alt.Chart(matches_over_time).mark_bar().encode(
        x='date:T',
        y='Matches:Q',
        tooltip=['date:T','Matches:Q']
    ).properties(width='container', height=300)
    st.altair_chart(chart, use_container_width=True)

# =========================
#    TAB: EXTENDED STATS
# =========================
with tab_extensions:
    st.header("Extended Stats & Fun Analyses")

    # === 1) Display longest winning/losing streak (filtered data) ===
    st.subheader("Longest Winning/Losing Streaks per Player")

    df_sorted = df_filtered.sort_values(
        ['date', 'match_number_total', 'match_number_day'], 
        ascending=True, na_position='last'
    )

    def compute_longest_win_streak(df_input, player):
        df_player = df_input[(df_input['Player1'] == player) | (df_input['Player2'] == player)]
        df_player = df_player.sort_values(['date', 'match_number_total', 'match_number_day'],
                                          ascending=True, na_position='last')

        longest = 0
        current = 0

        for _, row in df_player.iterrows():
            if row['Winner'] == player:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    def compute_longest_loss_streak(df_input, player):
        df_player = df_input[(df_input['Player1'] == player) | (df_input['Player2'] == player)]
        df_player = df_player.sort_values(['date', 'match_number_total', 'match_number_day'],
                                          ascending=True, na_position='last')

        longest = 0
        current = 0

        for _, row in df_player.iterrows():
            if row['Loser'] == player:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    # Compute for all players in the filter
    streaks = []
    unique_players_in_filtered = sorted(set(df_filtered['Player1']) | set(df_filtered['Player2']))
    for p in unique_players_in_filtered:
        w_streak = compute_longest_win_streak(df_sorted, p)
        l_streak = compute_longest_loss_streak(df_sorted, p)
        streaks.append((p, w_streak, l_streak))

    df_streaks = pd.DataFrame(streaks, columns=['Player','Longest_Win_Streak','Longest_Loss_Streak'])
    df_streaks.sort_values('Longest_Win_Streak', ascending=False, inplace=True)
    df_streaks.reset_index(drop=True, inplace=True)

    st.dataframe(df_streaks, use_container_width=True)

    # === 2) Show distribution of match results (Filtered) ===
    st.subheader("Distribution of Match Results (Filtered)")

    def normalize_result(row):
        """We invert (max:min) to unify 11:9 vs. 9:11 
           But you previously wanted 9:11 distinct from 11:9.
           So if you still want them unified, keep max/min;
           otherwise remove or adapt. We'll keep your version that unifies it.
        """
        s1, s2 = row['Score1'], row['Score2']
        mn, mx = int(min(s1, s2)), int(max(s1, s2))
        return f"{mx}:{mn}"

    df_filtered['ResultPair'] = df_filtered.apply(normalize_result, axis=1)
    pair_counts = df_filtered['ResultPair'].value_counts().reset_index()
    pair_counts.columns = ['ResultPair', 'Count']

    results_chart = alt.Chart(pair_counts).mark_bar().encode(
        x=alt.X('Count:Q', title='Number of Matches'),
        y=alt.Y('ResultPair:N', sort='-x', title='Score Category'),
        tooltip=['ResultPair','Count']
    ).properties(width='container', height=500)

    st.altair_chart(results_chart, use_container_width=True)

    # === 3) Identify the 'closest' matches (smallest margin) ===
    st.subheader("Closest Matches (Filtered)")

    n_closest = st.slider("Number of closest matches to display", min_value=1, max_value=20, value=10)
    df_filtered['TotalPoints'] = df_filtered['Score1'] + df_filtered['Score2']

    # Sort by margin ascending, then by total points descending to break ties
    # so that if multiple matches have the same margin, the highest total points appear first
    # Then pick the top n
    df_closest_sorted = df_filtered.sort_values(['PointDiff','TotalPoints'], ascending=[True, False])
    closest_subset = df_closest_sorted.head(n_closest)

    st.dataframe(
        closest_subset[[
            'match_number_total','date','Player1','Score1','Player2','Score2','PointDiff','TotalPoints'
        ]].reset_index(drop=True),
        use_container_width=True
    )

    # === 4) Calculate and display Elo ratings for all matches in chronological order ===
    st.subheader("Elo Ratings (Entire Dataset)")

    st.markdown("""
    The Elo system is a method for calculating the relative skill levels of players.  
    **Assumptions**:  
    - Initial Elo for every new player = 1500  
    - K-factor = 20 (determines rating adjustment speed)  
    - Sorted by date, then match_number_total (and match_number_day).  
    """)

    df_all_sorted = df.sort_values(['date','match_number_total','match_number_day'], ascending=True, na_position='last')
    elo_ratings = defaultdict(lambda: 1500)
    K = 20

    for _, row in df_all_sorted.iterrows():
        p1, p2 = row['Player1'], row['Player2']
        r1, r2 = elo_ratings[p1], elo_ratings[p2]

        # Expected scores
        exp1 = 1 / (1 + 10 ** ((r2 - r1)/400))
        exp2 = 1 / (1 + 10 ** ((r1 - r2)/400))

        # Actual
        if row['Score1'] > row['Score2']:
            a1, a2 = 1, 0
        else:
            a1, a2 = 0, 1

        # Update
        elo_ratings[p1] = r1 + K * (a1 - exp1)
        elo_ratings[p2] = r2 + K * (a2 - exp2)

    df_elo = pd.DataFrame([(p, rating) for p, rating in elo_ratings.items()],
                          columns=['Player','Elo_Rating'])
    df_elo.sort_values('Elo_Rating', ascending=False, inplace=True, ignore_index=True)

    st.dataframe(
        df_elo.style.format({"Elo_Rating":"{:.2f}"}),
        use_container_width=True
    )

    st.markdown("""
    **Note**: This Elo rating is computed across the *entire* dataset in chronological order, 
    regardless of the filtering above.
    """)

   # === 5) Performance by "Nth Match of Day" (the meltdown approach) ===
    st.subheader("Performance by Nth Match of Day")
    
    def meltdown_day_matches(df_in):
        """
        For each row (one match), create two rows:
         - one for Player1
         - one for Player2
        Then we group by (date, player) to find how many matches they've
        played that day (1st, 2nd, 3rd, etc.).
        We'll also track whether they won or not (did_win).
        """
        # Sort to keep chronological
        df_in = df_in.sort_values(['date','match_number_total','match_number_day'], ascending=True)
    
        # Subset for Player1
        df_p1 = df_in[['date','Player1','Winner','Loser','Score1','Score2','match_number_total','match_number_day']]
        df_p1 = df_p1.rename(columns={
            'Player1':'player',
            'Score1':'score_for_this_player',
            'Score2':'score_for_opponent'
        })
        df_p1['did_win'] = (df_p1['player'] == df_p1['Winner']).astype(int)
    
        # Subset for Player2
        df_p2 = df_in[['date','Player2','Winner','Loser','Score1','Score2','match_number_total','match_number_day']]
        df_p2 = df_p2.rename(columns={
            'Player2':'player',
            'Score2':'score_for_this_player',
            'Score1':'score_for_opponent'
        })
        df_p2['did_win'] = (df_p2['player'] == df_p2['Winner']).astype(int)
    
        # Combine
        df_stacked = pd.concat([df_p1, df_p2], ignore_index=True)
    
        # Group by date+player to find "which nth match of day"
        df_stacked = df_stacked.sort_values(['date','player','match_number_total','match_number_day'])
        df_stacked['MatchOfDay'] = df_stacked.groupby(['date','player']).cumcount() + 1
    
        return df_stacked
    
    # Melt down the FILTERED data
    df_daycount = meltdown_day_matches(df_filtered)
    
    # We'll group by (player, MatchOfDay) to see how many matches and how many wins
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
        default=available_players  # or pick some subset
    )
    
    if players_for_nth_chart:
        df_day_agg_display = df_day_agg[df_day_agg['player'].isin(players_for_nth_chart)]
        # We'll treat MatchOfDay as numeric so the regression can work.
        # If some players have only 1 data point, the regression won't show meaningfully; that's okay.
    
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
    
        # 1) The actual data lines with points
        lines_layer = base.mark_line(point=True)
    
        # 2) The linear regression trend line for each player
        trend_layer = (
            base
            .transform_regression(
                'MatchOfDay', 'win_rate', groupby=['player']
            )
            .mark_line(strokeDash=[4,4])   # dashed line for the trend
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
    The **solid line** represents the actual data points, and the **dashed line** is a **linear trend** for each player's performance.
    """)
