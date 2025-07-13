import streamlit as st
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection
import altair as alt
import plotly.graph_objects as go

# Import all our new modules
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
    simulate_tournament_bracket,
    calculate_performance_trajectory,
    what_if_scenario_analysis
)

from enhanced_visualizations import (
    create_player_comparison_radar,
    create_activity_heatmap,
    create_win_rate_opponent_heatmap,
    create_calendar_view,
    create_point_flow_sankey,
    create_performance_scatter,
    create_score_distribution_boxplot,
    create_bubble_performance_chart,
    create_sparklines_dataframe,
    add_milestone_annotations,
    create_momentum_indicator,
    create_rating_trend_indicator
)

from rating_utils import generate_elo_ratings_over_time
from color_palette import PRIMARY, SECONDARY, TERTIARY

# Page config
st.set_page_config(page_title="Ultimate Analytics Dashboard", layout="wide", page_icon="🚀")

@st.cache_data
def load_data():
    """Load and preprocess match data"""
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(worksheet="match_results")
    
    # Data preprocessing
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["Player1"] = df["Player1"].replace("Friede", "Friedemann")
    df["Player2"] = df["Player2"].replace("Friede", "Friedemann")
    
    # Add derived columns
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
    
    return df

@st.cache_data
def calculate_all_metrics(df):
    """Calculate all advanced metrics"""
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

def main():
    st.title("🚀 Ultimate Squash Analytics Dashboard")
    st.markdown("*Where data meets dominance on the court*")
    
    # Load data
    df = load_data()
    metrics = calculate_all_metrics(df)
    
    # Sidebar filters
    st.sidebar.header("🎯 Filters")
    
    # Date range
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Player filter
    all_players = sorted(set(df["Player1"]) | set(df["Player2"]))
    selected_players = st.sidebar.multiselect(
        "Players",
        all_players,
        default=all_players
    )
    
    # Apply filters
    df_filtered = df[
        (df["date"] >= pd.to_datetime(date_range[0])) &
        (df["date"] <= pd.to_datetime(date_range[1])) &
        (df["Player1"].isin(selected_players)) &
        (df["Player2"].isin(selected_players))
    ]
    
    # Main tabs
    tabs = st.tabs([
        "📊 Performance Overview",
        "🎯 Player Comparison",
        "🔮 Predictive Analytics",
        "💪 Performance Patterns",
        "🧠 Psychological Insights",
        "📈 Advanced Visualizations",
        "🏆 Tournament Simulator",
        "⚡ Real-time Dashboard"
    ])
    
    # Tab 1: Performance Overview
    with tabs[0]:
        st.header("Performance Overview")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(df_filtered))
            st.metric("Active Players", len(selected_players))
        
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
    
    # Tab 2: Player Comparison
    with tabs[1]:
        st.header("Player Comparison Dashboard")
        
        # Player selection
        comparison_players = st.multiselect(
            "Select players to compare (2-4 players)",
            all_players,
            default=all_players[:min(3, len(all_players))],
            max_selections=4
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
                
                # Add metrics from each dataframe
                for metric_name, metric_df in metrics.items():
                    if isinstance(metric_df, pd.DataFrame) and 'Player' in metric_df.columns:
                        player_metrics = metric_df[metric_df['Player'] == player]
                        if not player_metrics.empty:
                            if metric_name == 'clutch':
                                player_data['Clutch Rating'] = f"{player_metrics.iloc[0]['Clutch Rating']:.2f}"
                            elif metric_name == 'dominance':
                                player_data['Dominance'] = f"{player_metrics.iloc[0]['Dominance Score']:.1f}"
                            elif metric_name == 'consistency':
                                player_data['Consistency'] = f"{player_metrics.iloc[0]['Consistency Rating']:.1f}%"
                            elif metric_name == 'momentum':
                                player_data['Momentum'] = player_metrics.iloc[0]['Hot Streak']
                
                comparison_data.append(player_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    # Tab 3: Predictive Analytics
    with tabs[2]:
        st.header("🔮 Predictive Analytics & Match Forecasting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Match Predictor")
            pred_p1 = st.selectbox("Player 1", all_players, key="pred_p1")
            pred_p2 = st.selectbox("Player 2", [p for p in all_players if p != pred_p1], key="pred_p2")
            
            if st.button("Calculate Match Probability"):
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
        if not upset_df.empty:
            st.dataframe(
                upset_df.head(10).style.format({
                    'Favorite Win %': '{:.1f}%',
                    'Upset Probability': '{:.1f}%',
                    'Underdog Form': '{:.2f}',
                    'Favorite Struggles': '{:.2f}',
                    'Upset Index': '{:.1f}'
                }),
                use_container_width=True
            )
        
        # Performance Trajectory
        st.subheader("📈 Performance Trajectory Analysis")
        traj_player = st.selectbox("Select player for trajectory analysis", all_players)
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
    
    # Tab 4: Performance Patterns
    with tabs[3]:
        st.header("💪 Performance Patterns & Analytics")
        
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
    
    # Tab 5: Psychological Insights
    with tabs[4]:
        st.header("🧠 Psychological Insights & Mental Game")
        
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
            selected_player_nemesis = st.selectbox("Select player for nemesis analysis", all_players)
            
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
    
    # Tab 6: Advanced Visualizations
    with tabs[5]:
        st.header("📈 Advanced Visualizations")
        
        viz_tabs = st.tabs(["Activity Patterns", "Score Analysis", "Calendar View", "Match Flow"])
        
        with viz_tabs[0]:
            st.subheader("📅 Activity Patterns")
            activity_heatmap = create_activity_heatmap(df_filtered)
            st.altair_chart(activity_heatmap, use_container_width=True)
        
        with viz_tabs[1]:
            st.subheader("📊 Score Distribution Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                scatter_chart = create_performance_scatter(df_filtered)
                st.altair_chart(scatter_chart, use_container_width=True)
            
            with col2:
                boxplot = create_score_distribution_boxplot(df_filtered)
                st.altair_chart(boxplot, use_container_width=True)
        
        with viz_tabs[2]:
            st.subheader("📆 Match Calendar")
            cal_player = st.selectbox("Select player for calendar view (or leave empty for all)", 
                                     ["All"] + all_players)
            
            if cal_player == "All":
                calendar = create_calendar_view(df_filtered)
            else:
                calendar = create_calendar_view(df_filtered, cal_player)
            
            st.altair_chart(calendar, use_container_width=True)
        
        with viz_tabs[3]:
            st.subheader("🌊 Match Point Flow")
            recent_matches = df_filtered.sort_values('date', ascending=False).head(20)
            
            if not recent_matches.empty:
                match_options = []
                for _, match in recent_matches.iterrows():
                    match_options.append(
                        f"Match {match['match_number_total']}: {match['Player1']} vs {match['Player2']} "
                        f"({match['Score1']}-{match['Score2']}) - {match['date'].strftime('%Y-%m-%d')}"
                    )
                
                selected_match_str = st.selectbox("Select a match to visualize", match_options)
                selected_match_id = int(selected_match_str.split(':')[0].split()[1])
                
                sankey_fig = create_point_flow_sankey(df_filtered, selected_match_id)
                st.plotly_chart(sankey_fig, use_container_width=True)
    
    # Tab 7: Tournament Simulator
    with tabs[6]:
        st.header("🏆 Tournament Simulator")
        
        st.subheader("Configure Tournament")
        
        tournament_size = st.select_slider(
            "Tournament Size",
            options=[4, 8, 16],
            value=8
        )
        
        tournament_players = st.multiselect(
            f"Select {tournament_size} players",
            all_players,
            default=all_players[:min(tournament_size, len(all_players))]
        )
        
        if len(tournament_players) == tournament_size:
            num_sims = st.slider("Number of simulations", 100, 5000, 1000, 100)
            
            if st.button("Run Tournament Simulation"):
                with st.spinner("Running simulations..."):
                    results = simulate_tournament_bracket(
                        df_filtered, 
                        tournament_players,
                        metrics['elo_ratings'],
                        num_sims
                    )
                
                if 'error' not in results:
                    st.subheader("Tournament Predictions")
                    
                    # Results table
                    st.dataframe(
                        results['predictions'].style.format({
                            'Win %': '{:.1f}%',
                            'Final %': '{:.1f}%',
                            'Avg Wins': '{:.1f}'
                        }).background_gradient(subset=['Win %'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                    
                    # Most likely final
                    st.info(f"Most likely final: **{results['most_likely_final'][0]}** vs **{results['most_likely_final'][1]}**")
                else:
                    st.error(results['error'])
        else:
            st.warning(f"Please select exactly {tournament_size} players")
    
    # Tab 8: Real-time Dashboard
    with tabs[7]:
        st.header("⚡ Real-time Performance Dashboard")
        
        # Create a grid layout
        dashboard_player = st.selectbox("Select player for dashboard", all_players)
        
        if dashboard_player:
            # Player header with key stats
            col1, col2, col3, col4, col5 = st.columns(5)
            
            player_matches = df_filtered[
                (df_filtered["Player1"] == dashboard_player) | 
                (df_filtered["Player2"] == dashboard_player)
            ]
            
            wins = len(player_matches[player_matches["Winner"] == dashboard_player])
            total = len(player_matches)
            
            with col1:
                st.metric("Win Rate", f"{wins/total*100:.1f}%")
            
            with col2:
                if dashboard_player in metrics['elo_ratings']:
                    rating = metrics['elo_ratings'][dashboard_player]
                    st.metric("Elo Rating", f"{rating:.0f}")
            
            with col3:
                momentum_player = metrics['momentum'][metrics['momentum']['Player'] == dashboard_player]
                if not momentum_player.empty:
                    st.metric("Form", momentum_player.iloc[0]['Last 10 Form'])
            
            with col4:
                dominance_player = metrics['dominance'][metrics['dominance']['Player'] == dashboard_player]
                if not dominance_player.empty:
                    st.metric("Dominance", f"{dominance_player.iloc[0]['Dominance Score']:.1f}")
            
            with col5:
                consistency_player = metrics['consistency'][metrics['consistency']['Player'] == dashboard_player]
                if not consistency_player.empty:
                    st.metric("Consistency", f"{consistency_player.iloc[0]['Consistency Rating']:.1f}%")
            
            # Performance trends
            st.subheader("Performance Trends")
            
            # Get rating history for player
            player_elo_history = metrics['elo_history'][
                metrics['elo_history']['Player'] == dashboard_player
            ].copy()
            
            if not player_elo_history.empty:
                # Rating trend chart with milestones
                rating_chart = alt.Chart(player_elo_history).mark_line(
                    point=True,
                    strokeWidth=3
                ).encode(
                    x='date:T',
                    y='Elo Rating:Q',
                    tooltip=['date:T', 'Elo Rating:Q']
                ).properties(
                    title=f'{dashboard_player} Rating Progression',
                    height=300
                )
                
                # Add milestone annotations
                rating_chart = add_milestone_annotations(rating_chart, df_filtered)
                
                st.altair_chart(rating_chart, use_container_width=True)
            
            # What-if scenario analyzer
            st.subheader("What-If Scenario Analyzer")
            
            col1, col2 = st.columns(2)
            with col1:
                scenario_wins = st.number_input("Hypothetical Wins", 0, 20, 5)
            with col2:
                scenario_losses = st.number_input("Hypothetical Losses", 0, 20, 2)
            
            if st.button("Calculate Impact"):
                scenario_result = what_if_scenario_analysis(
                    df_filtered,
                    dashboard_player,
                    scenario_wins,
                    scenario_losses,
                    metrics['elo_ratings']
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Record Change", 
                             f"{scenario_result['current_record']} → {scenario_result['new_record']}")
                with col2:
                    st.metric("Win Rate Change",
                             f"{scenario_result['win_rate_change']*100:+.1f}%",
                             delta=f"{scenario_result['win_rate_change']*100:.1f}%")
                with col3:
                    st.metric("Rating Change",
                             f"{scenario_result['rating_change']:+.0f}",
                             delta=f"{scenario_result['rating_change']:.0f}")

if __name__ == "__main__":
    main()