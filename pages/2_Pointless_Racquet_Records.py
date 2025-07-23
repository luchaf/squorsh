import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import date
from color_palette import PRIMARY, SECONDARY, TERTIARY


# Create GSheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Session selection in sidebar
st.sidebar.header("Session Management")

# Get current session from session state
current_session = st.session_state.get("current_session", "")

# If no session is selected, try to load the active session from Google Sheets
if not current_session:
    try:
        sessions_df = conn.read(worksheet="sessions")
        if not sessions_df.empty and "session_name" in sessions_df.columns:
            # Check if there's an active session
            if "active" in sessions_df.columns:
                active_sessions = sessions_df[sessions_df["active"] == True]
                if not active_sessions.empty:
                    current_session = active_sessions.iloc[0]["session_name"]
                    st.session_state["current_session"] = current_session
                    st.sidebar.success(f"✅ Loaded active session: **{current_session}**")
                else:
                    # No active session set, auto-select first available
                    available_sessions = sessions_df["session_name"].dropna().tolist()
                    if available_sessions:
                        current_session = available_sessions[0]
                        st.session_state["current_session"] = current_session
                        st.sidebar.info(f"🔄 Auto-selected session: **{current_session}**")
                        st.sidebar.info("🔧 Go to Settings to set an active session for all devices")
                    else:
                        st.sidebar.error("❌ No sessions found!")
                        st.sidebar.markdown("**Please go to Settings to create a session:**")
                        if st.sidebar.button("🔧 Go to Settings"):
                            st.switch_page("pages/3_Settings.py")
                        st.stop()
            else:
                # No active column, auto-select first available
                available_sessions = sessions_df["session_name"].dropna().tolist()
                if available_sessions:
                    current_session = available_sessions[0]
                    st.session_state["current_session"] = current_session
                    st.sidebar.info(f"🔄 Auto-selected session: **{current_session}**")
                    st.sidebar.info("🔧 Go to Settings to set an active session for all devices")
                else:
                    st.sidebar.error("❌ No sessions found!")
                    st.sidebar.markdown("**Please go to Settings to create a session:**")
                    if st.sidebar.button("🔧 Go to Settings"):
                        st.switch_page("pages/3_Settings.py")
                    st.stop()
        else:
            # Sessions worksheet exists but has no data
            st.sidebar.error("❌ No sessions found!")
            st.sidebar.markdown("**Please go to Settings to create a session:**")
            if st.sidebar.button("🔧 Go to Settings"):
                st.switch_page("pages/3_Settings.py")
            st.stop()
    except Exception:
        # Sessions worksheet doesn't exist
        st.sidebar.error("❌ No sessions found!")
        st.sidebar.markdown("**Please go to Settings to create your first session:**")
        if st.sidebar.button("🔧 Go to Settings"):
            st.switch_page("pages/3_Settings.py")
        st.stop()

# Show current session status
st.sidebar.success(f"✅ Active Session: **{current_session}**")
if st.sidebar.button("🔧 Change Session"):
    st.switch_page("pages/3_Settings.py")

# Determine worksheet names based on current session
worksheet_name = f"{current_session}_match_results"
player_names_worksheet = f"{current_session}_player_names"

try:
    df = conn.read(worksheet=worksheet_name)
    
    # Handle empty worksheet (this is normal for new sessions)
    if df.empty:
        st.info(f"📋 Session worksheet '{worksheet_name}' is empty. This is normal for a new session - you can start entering match results below!")
        # Create an empty dataframe with the expected columns
        df = pd.DataFrame(columns=["date", "Player1", "Player2", "Score1", "Score2", "match_number_total", "match_number_day"])
            
except Exception as e:
    st.info(f"📋 Session worksheet '{worksheet_name}' doesn't exist yet. This is normal for a new session - you can start entering match results below!")
    # Create an empty dataframe with the expected columns
    df = pd.DataFrame(columns=["date", "Player1", "Player2", "Score1", "Score2", "match_number_total", "match_number_day"])

# Only process data if dataframe is not empty
if not df.empty:
    df["date"] = df["date"].astype(int).astype(str)
    for i in ["match_number_total", "match_number_day", "Score1", "Score2"]:
        df[i] = df[i].astype(int)

try:
    player_names_df = conn.read(worksheet=player_names_worksheet)
    if player_names_df.empty or "player_names" not in player_names_df.columns:
        st.warning(f"No player names found in worksheet '{player_names_worksheet}'. Using players from match data.")
        player_names = sorted(list(set(df["Player1"]) | set(df["Player2"])))
    else:
        player_names = player_names_df["player_names"].tolist()
except Exception as e:
    st.warning(f"Error loading player names: {str(e)}. Using players from match data.")
    player_names = sorted(list(set(df["Player1"]) | set(df["Player2"])))

(
    online_form,
    show_me_the_list,
    rankings,
) = st.tabs(["📝 Log Results", "🏓 Recent Matches", "🏆 Rankings"])


with online_form:

    def reset_session_state():
        """Helper function to reset session state."""
        if player_names:
            st.session_state["player1_name"] = player_names[0]  # Default to the first player
            st.session_state["player2_name"] = player_names[min(3, len(player_names) - 1)]  # Default to the fourth player or last available
        else:
            st.session_state["player1_name"] = "Player 1"
            st.session_state["player2_name"] = "Player 2"
        st.session_state["player1_score"] = 0
        st.session_state["player2_score"] = 0
        st.session_state["matchday_input"] = date.today()
        st.session_state["data_written"] = False

    def display_enter_match_results(df):
        # Initialize session state variables if not already set
        if "data_written" not in st.session_state:
            st.session_state["data_written"] = False
        if "player1_name" not in st.session_state:
            if player_names:
                st.session_state["player1_name"] = player_names[0]  # Default to the first player
            else:
                st.session_state["player1_name"] = "Player 1"
        if "player1_score" not in st.session_state:
            st.session_state["player1_score"] = 0
        if "player2_name" not in st.session_state:
            if player_names:
                st.session_state["player2_name"] = player_names[min(3, len(player_names) - 1)]  # Default to the fourth player or last available
            else:
                st.session_state["player2_name"] = "Player 2"
        if "player2_score" not in st.session_state:
            st.session_state["player2_score"] = 0
        if "matchday_input" not in st.session_state:
            st.session_state["matchday_input"] = date.today()

        if st.session_state["data_written"]:
            # Display the success message and allow adding another match
            st.success("Match result saved! Enter another?")
            if st.button("Add Another Match"):
                reset_session_state()
                st.rerun()  # Rerun to reset the form
        else:
            # Display the form to log match results
            st.title("Log Your Match Results")

            # Widgets with session state management
            if player_names:
                player1_name = st.selectbox(
                    "Player 1",
                    player_names,
                    index=player_names.index(st.session_state["player1_name"]) if st.session_state["player1_name"] in player_names else 0,
                    key="player1_name",
                )
            else:
                player1_name = st.text_input(
                    "Player 1",
                    value=st.session_state["player1_name"],
                    key="player1_name",
                )
            player1_score = st.number_input(
                "Player 1 Score",
                min_value=0,
                step=1,
                value=st.session_state["player1_score"],
                key="player1_score",
            )
            if player_names:
                player2_name = st.selectbox(
                    "Player 2",
                    player_names,
                    index=player_names.index(st.session_state["player2_name"]) if st.session_state["player2_name"] in player_names else 0,
                    key="player2_name",
                )
            else:
                player2_name = st.text_input(
                    "Player 2",
                    value=st.session_state["player2_name"],
                    key="player2_name",
                )
            player2_score = st.number_input(
                "Player 2 Score",
                min_value=0,
                step=1,
                value=st.session_state["player2_score"],
                key="player2_score",
            )
            matchday_input = st.date_input(
                "Matchday",
                value=st.session_state["matchday_input"],
                key="matchday_input",
            )

            # Submit button to save the match result
            if st.button("Submit Match Result"):
                # Calculate match_number_total
                if not df.empty:
                    match_number_total = df["match_number_total"].max() + 1
                else:
                    match_number_total = 1

                # Calculate match_number_day
                current_date = matchday_input.strftime("%Y%m%d")
                if current_date in df["date"].values:
                    match_number_day = (
                        df[df["date"] == current_date]["match_number_day"].max() + 1
                    )
                else:
                    match_number_day = 1

                new_data = {
                    "Player1": player1_name,
                    "Score1": player1_score,
                    "Player2": player2_name,
                    "Score2": player2_score,
                    "date": current_date,
                    "match_number_total": match_number_total,
                    "match_number_day": match_number_day,
                }

                updated_df = pd.concat(
                    [df, pd.DataFrame([new_data])], ignore_index=True
                )
                # Update worksheet
                conn.update(worksheet=worksheet_name, data=updated_df)
                st.cache_data.clear()
                st.session_state["data_written"] = True
                st.rerun()  # Rerun to show the success message

    display_enter_match_results(df)

with show_me_the_list:
    st.header("🏓 Match Results Management")
    st.markdown(f"**Current Session:** {current_session}")
    st.markdown(f"**Managing matches for:** `{worksheet_name}`")
    
    # Display current matches
    st.subheader("Current Matches")
    if not df.empty:
        # Show statistics above the table
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Matches:** {len(df)}")
        with col2:
            if not df.empty:
                latest_date = pd.to_datetime(df["date"], format="%Y%m%d").max()
                st.write(f"**Latest Match:** {latest_date.strftime('%Y-%m-%d')}")
            else:
                st.write("**Latest Match:** None")
        
        st.divider()
        
        # Create a formatted display of matches
        display_df = df.copy()
        # Format date for better readability
        display_df["Date"] = pd.to_datetime(display_df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        # Create a match description column
        display_df["Match"] = display_df.apply(
            lambda row: f"{row['Player1']} ({row['Score1']}) vs {row['Player2']} ({row['Score2']})", axis=1
        )
        # Select and reorder columns for display
        display_columns = ["match_number_total", "Date", "Match", "match_number_day"]
        display_df = display_df[display_columns]
        display_df.columns = ["Match #", "Date", "Match Result", "Day #"]
        display_df.index = display_df.index + 1  # Start numbering from 1
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No matches found. Add some matches in the 'Online form' tab!")
    
    
with rankings:
    st.header("🏆 Rankings & Statistics")
    st.markdown(f"**Current Session:** {current_session}")
    
    if df.empty:
        st.info("No matches found. Play some matches to see rankings!")
    else:
        # Calculate player statistics
        player_stats = {}
        
        # Process each match to gather statistics
        for _, match in df.iterrows():
            player1 = match['Player1']
            player2 = match['Player2']
            score1 = int(match['Score1'])
            score2 = int(match['Score2'])
            
            # Initialize players if not exists
            if player1 not in player_stats:
                player_stats[player1] = {'games': 0, 'wins': 0, 'total_points': 0}
            if player2 not in player_stats:
                player_stats[player2] = {'games': 0, 'wins': 0, 'total_points': 0}
            
            # Update statistics
            player_stats[player1]['games'] += 1
            player_stats[player2]['games'] += 1
            player_stats[player1]['total_points'] += score1
            player_stats[player2]['total_points'] += score2
            
            # Determine winner
            if score1 > score2:
                player_stats[player1]['wins'] += 1
            elif score2 > score1:
                player_stats[player2]['wins'] += 1
        
        # Create rankings dataframe
        rankings_data = []
        for player, stats in player_stats.items():
            games = stats['games']
            wins = stats['wins']
            total_points = stats['total_points']
            
            # Calculate metrics
            avg_points = total_points / games if games > 0 else 0
            win_rate = wins / games if games > 0 else 0
            multiplier = 1 + 2 * (games / (games + 5)) if games > 0 else 1
            
            # Score formula: (P+(2*S/SP))*(1+(SP/(SP+5)))
            if games > 0:
                score = (avg_points + (2 * win_rate)) * multiplier
            else:
                score = 0
            
            rankings_data.append({
                'Player': player,
                'Matches': games,
                'Win Rate': round(win_rate * 100, 1),  # Convert to percentage
                'Avg Points': round(avg_points, 1),
                'Multiplier': round(multiplier, 3),
                'Score': round(score, 2)
            })
        
        # Sort by score (descending)
        rankings_data.sort(key=lambda x: x['Score'], reverse=True)
        
        # Create dataframe for display
        rankings_df = pd.DataFrame(rankings_data)
        
        # Add rank column
        rankings_df.insert(0, 'Rank', range(1, len(rankings_df) + 1))
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Total Players:** {len(rankings_df)}")
        with col2:
            st.write(f"**Total Matches:** {len(df)}")
        with col3:
            if not rankings_df.empty:
                top_player = rankings_df.iloc[0]['Player']
                st.write(f"**Top Player:** {top_player}")
        
        st.divider()
        
        # Display rankings table
        st.subheader("Player Rankings")
        st.markdown("""
        **Legend:**
        - **Matches**: Total games played
        - **Win Rate**: Percentage of games won
        - **Avg Points**: Average points scored per game
        - **Multiplier**: Activity bonus = (1+2x(Matches/(Matches+5)))
        - **Score**: Ranking Score = (Avg Points + (2×Win Rate/100)) × Multiplier
        """)
        
        # Style the dataframe
        styled_df = rankings_df.style.format({
            'Win Rate': '{:.1f}%',
            'Avg Points': '{:.1f}',
            'Multiplier': '{:.3f}',
            'Score': '{:.2f}'
        }).background_gradient(
            subset=['Score'], 
            cmap='RdYlGn',
            vmin=rankings_df['Score'].min(),
            vmax=rankings_df['Score'].max()
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Additional insights
        if len(rankings_df) > 1:
            st.divider()
            st.subheader("Quick Insights")
            
            col1, col2 = st.columns(2)
            with col1:
                # Most active player
                most_active = rankings_df.loc[rankings_df['Matches'].idxmax()]
                st.info(f"🏃 **Most Active:** {most_active['Player']} ({most_active['Matches']} games)")
                
                # Highest average points
                highest_avg = rankings_df.loc[rankings_df['Avg Points'].idxmax()]
                st.info(f"🎯 **Highest Avg Points:** {highest_avg['Player']} ({highest_avg['Avg Points']} pts/game)")
            
            with col2:
                # Best win rate (for players with at least 3 games)
                qualified_players = rankings_df[rankings_df['Matches'] >= 3]
                if not qualified_players.empty:
                    best_win_rate = qualified_players.loc[qualified_players['Win Rate'].idxmax()]
                    st.info(f"📈 **Best Win Rate:** {best_win_rate['Player']} ({best_win_rate['Win Rate']}%)")
                else:
                    st.info("📈 **Best Win Rate:** Need 3+ games")
                
                # Highest multiplier (most experienced)
                highest_mult = rankings_df.loc[rankings_df['Multiplier'].idxmax()]
                st.info(f"⬆️ **Highest Multiplier:** {highest_mult['Player']} ({highest_mult['Multiplier']})")
