import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import calendar
from color_palette import SEQUENCE, HEATMAP_SCHEME, PRIMARY, SECONDARY


def create_player_comparison_radar(
    df: pd.DataFrame,
    players: List[str],
    metrics_df: Dict[str, pd.DataFrame]
) -> go.Figure:
    """Create interactive radar chart comparing multiple players"""
    
    categories = [
        'Win Rate',
        'Dominance Score',
        'Clutch Performance',
        'Consistency',
        'Momentum',
        'Mental Toughness'
    ]
    
    fig = go.Figure()
    
    for i, player in enumerate(players):
        # Gather metrics for each player
        values = []
        
        # Win Rate
        player_matches = df[(df["Player1"] == player) | (df["Player2"] == player)]
        wins = len(player_matches[player_matches["Winner"] == player])
        win_rate = (wins / len(player_matches) * 100) if len(player_matches) > 0 else 0
        values.append(win_rate)
        
        # Dominance Score (normalized to 0-100)
        if ('dominance' in metrics_df and 
            not metrics_df['dominance'].empty and 
            'Player' in metrics_df['dominance'].columns and 
            player in metrics_df['dominance']['Player'].values):
            dom_score = metrics_df['dominance'][metrics_df['dominance']['Player'] == player]['Dominance Score'].iloc[0]
            values.append(min(100, max(0, dom_score)))
        else:
            values.append(50)
        
        # Clutch Performance
        if ('clutch' in metrics_df and 
            not metrics_df['clutch'].empty and 
            'Player' in metrics_df['clutch'].columns and 
            player in metrics_df['clutch']['Player'].values):
            clutch_rating = metrics_df['clutch'][metrics_df['clutch']['Player'] == player]['Clutch Rating'].iloc[0]
            values.append(min(100, clutch_rating * 100))
        else:
            values.append(50)
        
        # Consistency
        if ('consistency' in metrics_df and 
            not metrics_df['consistency'].empty and 
            'Player' in metrics_df['consistency'].columns and 
            player in metrics_df['consistency']['Player'].values):
            consistency = metrics_df['consistency'][metrics_df['consistency']['Player'] == player]['Consistency Rating'].iloc[0]
            values.append(consistency)
        else:
            values.append(50)
        
        # Momentum
        if ('momentum' in metrics_df and 
            not metrics_df['momentum'].empty and 
            'Player' in metrics_df['momentum'].columns and 
            player in metrics_df['momentum']['Player'].values):
            momentum = metrics_df['momentum'][metrics_df['momentum']['Player'] == player]['Momentum Score'].iloc[0]
            values.append(momentum)
        else:
            values.append(50)
        
        # Mental Toughness
        if ('pressure' in metrics_df and 
            not metrics_df['pressure'].empty and 
            'Player' in metrics_df['pressure'].columns and 
            player in metrics_df['pressure']['Player'].values):
            mental = metrics_df['pressure'][metrics_df['pressure']['Player'] == player]['Mental Fortitude'].iloc[0]
            values.append(mental)
        else:
            values.append(50)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=player,
            line_color=SEQUENCE[i % len(SEQUENCE)],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Player Comparison Radar",
        height=500
    )
    
    return fig


def create_activity_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Create a heatmap showing playing frequency by day and hour"""
    
    # Extract day of week and hour (would need timestamp data for hour)
    # For now, we'll do day of week vs week of year
    df_heat = df.copy()
    df_heat['day_of_week'] = df_heat['date'].dt.day_name()
    df_heat['week_of_year'] = df_heat['date'].dt.isocalendar().week
    df_heat['year'] = df_heat['date'].dt.year
    
    # Count matches per day/week
    heat_data = df_heat.groupby(['year', 'week_of_year', 'day_of_week']).size().reset_index(name='matches')
    
    # Order days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    heatmap = alt.Chart(heat_data).mark_rect().encode(
        x=alt.X('week_of_year:O', title='Week of Year'),
        y=alt.Y('day_of_week:N', sort=day_order, title='Day of Week'),
        color=alt.Color('matches:Q', 
                       scale=alt.Scale(scheme=HEATMAP_SCHEME),
                       title='Matches'),
        tooltip=['year:N', 'week_of_year:O', 'day_of_week:N', 'matches:Q']
    ).properties(
        title='Match Activity Heatmap',
        width=800,
        height=300
    ).facet(
        row='year:N'
    )
    
    return heatmap


def create_win_rate_opponent_heatmap(df: pd.DataFrame) -> alt.Chart:
    """Create heatmap of win rates by opponent and day of week"""
    
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    heatmap_data = []
    
    for player in players:
        for opponent in players:
            if player == opponent:
                continue
            
            h2h = df[
                ((df["Player1"] == player) & (df["Player2"] == opponent)) |
                ((df["Player1"] == opponent) & (df["Player2"] == player))
            ]
            
            if len(h2h) > 0:
                wins = len(h2h[h2h["Winner"] == player])
                win_rate = wins / len(h2h)
                
                heatmap_data.append({
                    "Player": player,
                    "Opponent": opponent,
                    "Win Rate": win_rate,
                    "Matches": len(h2h)
                })
    
    if not heatmap_data:
        return alt.Chart(pd.DataFrame()).mark_text().encode(
            text=alt.value("No head-to-head data available")
        )
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    heatmap = alt.Chart(heatmap_df).mark_rect().encode(
        x=alt.X('Opponent:N', title='Opponent'),
        y=alt.Y('Player:N', title='Player'),
        color=alt.Color('Win Rate:Q',
                       scale=alt.Scale(scheme='redblue', domain=[0, 1]),
                       title='Win Rate'),
        tooltip=['Player:N', 'Opponent:N', 
                alt.Tooltip('Win Rate:Q', format='.2%'),
                'Matches:Q']
    ).properties(
        title='Head-to-Head Win Rate Heatmap',
        width=500,
        height=500
    )
    
    return heatmap


def create_calendar_view(df: pd.DataFrame, player: str = None) -> alt.Chart:
    """Create calendar view with color-coded match results"""
    
    if player:
        player_df = df[(df["Player1"] == player) | (df["Player2"] == player)].copy()
        player_df['result'] = player_df['Winner'].apply(lambda x: 'Win' if x == player else 'Loss')
        player_df['color_value'] = player_df['result'].map({'Win': 1, 'Loss': -1})
        title = f"Match Calendar for {player}"
    else:
        player_df = df.copy()
        player_df['result'] = 'Match'
        player_df['color_value'] = 1
        title = "Match Calendar"
    
    # Add calendar components
    player_df['year'] = player_df['date'].dt.year
    player_df['month'] = player_df['date'].dt.month
    player_df['day'] = player_df['date'].dt.day
    player_df['day_of_week'] = player_df['date'].dt.dayofweek
    player_df['week'] = player_df['date'].dt.isocalendar().week
    
    # Create calendar chart
    calendar_chart = alt.Chart(player_df).mark_rect(
        cornerRadius=3
    ).encode(
        x=alt.X('day_of_week:O', 
               title=None,
               axis=alt.Axis(
                   labelExpr="['S', 'M', 'T', 'W', 'T', 'F', 'S'][datum.value]"
               )),
        y=alt.Y('week:O', title='Week'),
        color=alt.Color('color_value:Q',
                       scale=alt.Scale(
                           domain=[-1, 0, 1],
                           range=['#d62728', '#ffffff', '#2ca02c']
                       ),
                       legend=alt.Legend(
                           title='Result',
                           labelExpr="datum.value == 1 ? 'Win' : datum.value == -1 ? 'Loss' : 'No Match'"
                       )) if player else alt.Color('color_value:Q', legend=None),
        tooltip=['date:T', 'result:N'] + 
               (['Player1:N', 'Score1:Q', 'Player2:N', 'Score2:Q'] if not player else [])
    ).properties(
        title=title,
        width=600,
        height=150
    ).facet(
        column=alt.Column('month:O', 
                         title=None,
                         header=alt.Header(
                             labelExpr="['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][datum.value - 1]"
                         )),
        row='year:O'
    )
    
    return calendar_chart


def create_point_flow_sankey(df: pd.DataFrame, match_id: int) -> go.Figure:
    """Create Sankey diagram showing point flow in a match"""
    
    match = df[df["match_number_total"] == match_id].iloc[0]
    
    # Simulate point progression (would need actual point-by-point data)
    # This is a simplified visualization
    p1, p2 = match["Player1"], match["Player2"]
    s1, s2 = int(match["Score1"]), int(match["Score2"])
    
    # Create nodes
    labels = []
    for i in range(max(s1, s2) + 1):
        labels.append(f"{p1}: {i}")
        labels.append(f"{p2}: {i}")
    
    # Create links (simplified progression)
    source = []
    target = []
    value = []
    colors = []
    
    p1_score = 0
    p2_score = 0
    
    while p1_score < s1 or p2_score < s2:
        if p1_score < s1:
            source.append(labels.index(f"{p1}: {p1_score}"))
            target.append(labels.index(f"{p1}: {p1_score + 1}"))
            value.append(1)
            colors.append('rgba(31, 119, 180, 0.5)')
            p1_score += 1
        
        if p2_score < s2:
            source.append(labels.index(f"{p2}: {p2_score}"))
            target.append(labels.index(f"{p2}: {p2_score + 1}"))
            value.append(1)
            colors.append('rgba(255, 127, 14, 0.5)')
            p2_score += 1
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="rgba(31, 119, 180, 0.8)"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=colors
        )
    )])
    
    fig.update_layout(
        title_text=f"Point Flow: {p1} vs {p2} ({s1}-{s2})",
        font_size=10,
        height=400
    )
    
    return fig


def create_performance_scatter(df: pd.DataFrame) -> alt.Chart:
    """Create scatter plot of score differential vs match intensity"""
    
    scatter_data = df.copy()
    scatter_data['TotalPoints'] = scatter_data['Score1'] + scatter_data['Score2']
    scatter_data['ScoreDiff'] = abs(scatter_data['Score1'] - scatter_data['Score2'])
    
    scatter = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('TotalPoints:Q', 
               title='Total Points (Match Intensity)',
               scale=alt.Scale(domain=[0, 30])),
        y=alt.Y('ScoreDiff:Q', 
               title='Score Differential',
               scale=alt.Scale(domain=[0, 11])),
        color=alt.Color('Winner:N', title='Winner'),
        tooltip=['date:T', 'Player1:N', 'Score1:Q', 'Player2:N', 'Score2:Q']
    ).properties(
        title='Match Intensity vs Score Differential',
        width=600,
        height=400
    )
    
    # Add trend line
    trend = scatter.transform_regression(
        'TotalPoints', 'ScoreDiff'
    ).mark_line(color='red', strokeDash=[5, 5])
    
    return scatter + trend


def create_score_distribution_boxplot(df: pd.DataFrame) -> alt.Chart:
    """Create box plots for score distributions by player"""
    
    # Reshape data to have one row per player-match-score
    scores_data = []
    for _, match in df.iterrows():
        scores_data.append({
            'Player': match['Player1'],
            'Score': match['Score1'],
            'Result': 'Win' if match['Winner'] == match['Player1'] else 'Loss'
        })
        scores_data.append({
            'Player': match['Player2'],
            'Score': match['Score2'],
            'Result': 'Win' if match['Winner'] == match['Player2'] else 'Loss'
        })
    
    scores_df = pd.DataFrame(scores_data)
    
    boxplot = alt.Chart(scores_df).mark_boxplot(extent='min-max').encode(
        x=alt.X('Player:N', title='Player'),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 15])),
        color=alt.Color('Result:N', 
                       scale=alt.Scale(domain=['Win', 'Loss'],
                                     range=['#2ca02c', '#d62728']))
    ).properties(
        title='Score Distribution by Player and Result',
        width=600,
        height=400
    )
    
    return boxplot


def create_bubble_performance_chart(df: pd.DataFrame) -> alt.Chart:
    """Create bubble chart: wins (x), points (y), match count (size)"""
    
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    bubble_data = []
    
    for player in players:
        player_matches = df[(df["Player1"] == player) | (df["Player2"] == player)]
        wins = len(player_matches[player_matches["Winner"] == player])
        
        # Calculate total points
        points = 0
        for _, match in player_matches.iterrows():
            if match["Player1"] == player:
                points += match["Score1"]
            else:
                points += match["Score2"]
        
        bubble_data.append({
            'Player': player,
            'Wins': wins,
            'Points': points,
            'Matches': len(player_matches),
            'Win Rate': wins / len(player_matches) if len(player_matches) > 0 else 0
        })
    
    bubble_df = pd.DataFrame(bubble_data)
    
    bubble_chart = alt.Chart(bubble_df).mark_circle().encode(
        x=alt.X('Wins:Q', title='Total Wins'),
        y=alt.Y('Points:Q', title='Total Points'),
        size=alt.Size('Matches:Q', 
                     scale=alt.Scale(range=[100, 1000]),
                     title='Matches Played'),
        color=alt.Color('Win Rate:Q',
                       scale=alt.Scale(scheme='viridis'),
                       title='Win Rate'),
        tooltip=['Player:N', 'Wins:Q', 'Points:Q', 'Matches:Q',
                alt.Tooltip('Win Rate:Q', format='.2%')]
    ).properties(
        title='Player Performance Overview',
        width=600,
        height=400
    )
    
    # Add player labels
    text = alt.Chart(bubble_df).mark_text(
        align='center',
        baseline='middle',
        fontSize=10
    ).encode(
        x='Wins:Q',
        y='Points:Q',
        text='Player:N'
    )
    
    return bubble_chart + text


def create_sparklines_dataframe(df: pd.DataFrame, num_recent: int = 10) -> pd.DataFrame:
    """Create a dataframe with sparkline data for recent performance"""
    
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    sparkline_data = []
    
    for player in players:
        player_matches = df[
            (df["Player1"] == player) | (df["Player2"] == player)
        ].sort_values("date").tail(num_recent)
        
        if len(player_matches) == 0:
            continue
        
        # Get win/loss sequence
        results = []
        for _, match in player_matches.iterrows():
            results.append(1 if match["Winner"] == player else 0)
        
        # Create sparkline string
        sparkline = ''.join(['✓' if r else '✗' for r in results])
        
        # Calculate trend
        if len(results) >= 3:
            recent_rate = sum(results[-3:]) / 3
            older_rate = sum(results[:-3]) / len(results[:-3]) if len(results) > 3 else 0
            trend = '↑' if recent_rate > older_rate else '↓' if recent_rate < older_rate else '→'
        else:
            trend = '→'
        
        sparkline_data.append({
            'Player': player,
            'Recent Form': sparkline,
            'Trend': trend,
            'Last 10': f"{sum(results)}/{len(results)}",
            'Win %': sum(results) / len(results) * 100
        })
    
    return pd.DataFrame(sparkline_data).sort_values('Win %', ascending=False)


def add_milestone_annotations(chart: alt.Chart, df: pd.DataFrame) -> alt.Chart:
    """Add milestone markers to time series charts"""
    
    milestones = []
    
    # Find milestone matches
    for player in set(df["Player1"]) | set(df["Player2"]):
        player_wins = df[df["Winner"] == player].sort_values("date").reset_index(drop=True)
        
        # 50th, 100th wins
        for milestone_num in [50, 100, 150, 200]:
            if len(player_wins) >= milestone_num:
                milestone_match = player_wins.iloc[milestone_num - 1]
                milestones.append({
                    'date': milestone_match['date'],
                    'player': player,
                    'milestone': f"{milestone_num}th Win",
                    'y_position': milestone_num
                })
    
    if not milestones:
        return chart
    
    milestones_df = pd.DataFrame(milestones)
    
    # Add milestone markers
    milestone_chart = alt.Chart(milestones_df).mark_rule(
        strokeDash=[5, 5],
        color='gray'
    ).encode(
        x='date:T',
        tooltip=['date:T', 'player:N', 'milestone:N']
    )
    
    milestone_text = alt.Chart(milestones_df).mark_text(
        align='left',
        baseline='middle',
        dx=5,
        fontSize=10
    ).encode(
        x='date:T',
        y=alt.value(10),
        text='milestone:N'
    )
    
    return chart + milestone_chart + milestone_text


def create_momentum_indicator(momentum_score: float) -> str:
    """Create visual momentum indicator"""
    
    if momentum_score >= 80:
        return "🔥🔥🔥 ON FIRE!"
    elif momentum_score >= 60:
        return "🔥🔥 Hot Streak"
    elif momentum_score >= 40:
        return "➡️ Steady"
    elif momentum_score >= 20:
        return "❄️ Cooling Down"
    else:
        return "❄️❄️ Cold Streak"


def create_rating_trend_indicator(current: float, previous: float) -> str:
    """Create visual rating trend indicator"""
    
    change = current - previous
    if change > 50:
        return f"⬆️⬆️ +{change:.0f}"
    elif change > 20:
        return f"⬆️ +{change:.0f}"
    elif change > 0:
        return f"↗️ +{change:.0f}"
    elif change > -20:
        return f"↘️ {change:.0f}"
    elif change > -50:
        return f"⬇️ {change:.0f}"
    else:
        return f"⬇️⬇️ {change:.0f}"