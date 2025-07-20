import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import date
from color_palette import PRIMARY, SECONDARY, TERTIARY


# Create GSheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Mode selection in sidebar
st.sidebar.header("Data Source")
mode = st.sidebar.radio(
    "Select Mode",
    ["Season Mode", "Tournament Mode"],
    index=0,
    help="Season Mode uses regular match data, Tournament Mode uses tournament-specific data"
)

# Determine worksheet names based on mode
if mode == "Tournament Mode":
    worksheet_name = "match_results_tournament"
    player_names_worksheet = "player_names_tournament"
else:
    worksheet_name = "match_results"
    player_names_worksheet = "player_names"

try:
    df = conn.read(worksheet=worksheet_name)
    
    # Handle empty worksheet (this is normal for new tournaments)
    if df.empty:
        if mode == "Tournament Mode":
            st.info(f"📋 Tournament worksheet '{worksheet_name}' is empty. This is normal for a new tournament - you can start entering match results below!")
            # Create an empty dataframe with the expected columns for tournament mode
            df = pd.DataFrame(columns=["date", "Player1", "Player2", "Score1", "Score2", "match_number_total", "match_number_day"])
        else:
            st.error(f"No data found in worksheet '{worksheet_name}'. Please check if the worksheet exists and contains data.")
            st.stop()
            
except Exception as e:
    if mode == "Tournament Mode":
        st.info(f"📋 Tournament worksheet '{worksheet_name}' doesn't exist yet. This is normal for a new tournament - you can start entering match results below!")
        # Create an empty dataframe with the expected columns for tournament mode
        df = pd.DataFrame(columns=["date", "Player1", "Player2", "Score1", "Score2", "match_number_total", "match_number_day"])
    else:
        st.error(f"Error loading worksheet '{worksheet_name}': {str(e)}")
        st.stop()

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
    player_management,
) = st.tabs(["Online form", "List of recorded matches", "Player Management"])


with online_form:

    def reset_session_state():
        """Helper function to reset session state."""
        st.session_state["player1_name"] = player_names[
            0
        ]  # Default to the first player
        st.session_state["player1_score"] = 0
        st.session_state["player2_name"] = player_names[
            3
        ]  # Default to the first player
        st.session_state["player2_score"] = 0
        st.session_state["matchday_input"] = date.today()
        st.session_state["data_written"] = False

    def display_enter_match_results(df):
        # Initialize session state variables if not already set
        if "data_written" not in st.session_state:
            st.session_state["data_written"] = False
        if "player1_name" not in st.session_state:
            st.session_state["player1_name"] = player_names[
                0
            ]  # Default to the first player
        if "player1_score" not in st.session_state:
            st.session_state["player1_score"] = 0
        if "player2_name" not in st.session_state:
            st.session_state["player2_name"] = player_names[
                3
            ]  # Default to the first player
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
            player1_name = st.selectbox(
                "Player 1",
                player_names,
                index=player_names.index(st.session_state["player1_name"]),
                key="player1_name",
            )
            player1_score = st.number_input(
                "Player 1 Score",
                min_value=0,
                step=1,
                value=st.session_state["player1_score"],
                key="player1_score",
            )
            player2_name = st.selectbox(
                "Player 2",
                player_names,
                index=player_names.index(st.session_state["player2_name"]),
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
    st.dataframe(df)

with player_management:
    st.header("🎾 Player Management")
    st.markdown(f"**Current Mode:** {mode}")
    st.markdown(f"**Managing players for:** `{player_names_worksheet}`")
    
    # Display current players
    st.subheader("Current Players")
    if player_names:
        col1, col2 = st.columns([3, 1])
        with col1:
            # Create a nice display of current players
            players_df = pd.DataFrame({"Player Name": player_names})
            players_df.index = players_df.index + 1  # Start numbering from 1
            st.dataframe(players_df, use_container_width=True)
        with col2:
            st.metric("Total Players", len(player_names))
    else:
        st.info("No players found. Add some players below!")
    
    st.divider()
    
    # Add new player form
    st.subheader("Add New Player")
    
    with st.form("add_player_form"):
        new_player_name = st.text_input(
            "Player Name",
            placeholder="Enter player name...",
            help="Enter the full name of the new player"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("Add Player", type="primary")
        
        if submit_button:
            if new_player_name.strip():
                # Check if player already exists
                if new_player_name.strip() in player_names:
                    st.error(f"Player '{new_player_name}' already exists!")
                else:
                    try:
                        # Add new player to the list
                        updated_player_names = player_names + [new_player_name.strip()]
                        
                        # Create dataframe for updating
                        updated_players_df = pd.DataFrame({"player_names": updated_player_names})
                        
                        # Update the worksheet
                        conn.update(worksheet=player_names_worksheet, data=updated_players_df)
                        
                        # Clear cache and show success
                        st.cache_data.clear()
                        st.success(f"✅ Player '{new_player_name}' added successfully!")
                        st.info("🔄 Page will refresh to show the updated player list.")
                        st.rerun()
                        
                    except Exception as e:
                        if "Worksheet not found" in str(e) or "does not exist" in str(e):
                            st.error(f"Worksheet '{player_names_worksheet}' doesn't exist yet. This will be created automatically when you add the first player.")
                            try:
                                # Try to create the worksheet by updating with initial data
                                initial_players_df = pd.DataFrame({"player_names": [new_player_name.strip()]})
                                conn.update(worksheet=player_names_worksheet, data=initial_players_df)
                                st.cache_data.clear()
                                st.success(f"✅ Created worksheet '{player_names_worksheet}' and added player '{new_player_name}'!")
                                st.info("🔄 Page will refresh to show the updated player list.")
                                st.rerun()
                            except Exception as create_error:
                                st.error(f"Error creating worksheet: {str(create_error)}")
                        else:
                            st.error(f"Error adding player: {str(e)}")
            else:
                st.error("Please enter a valid player name.")
    
    # Bulk add players (optional)
    st.divider()
    st.subheader("Bulk Add Players")
    st.markdown("*Add multiple players at once (one per line)*")
    
    with st.form("bulk_add_players_form"):
        bulk_players = st.text_area(
            "Player Names (one per line)",
            placeholder="Player 1\nPlayer 2\nPlayer 3\n...",
            height=150,
            help="Enter multiple player names, one per line"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            bulk_submit = st.form_submit_button("Add All Players", type="secondary")
        
        if bulk_submit:
            if bulk_players.strip():
                # Parse the input
                new_players = [name.strip() for name in bulk_players.strip().split('\n') if name.strip()]
                
                if new_players:
                    # Check for duplicates
                    existing_players = set(player_names)
                    unique_new_players = [p for p in new_players if p not in existing_players]
                    duplicates = [p for p in new_players if p in existing_players]
                    
                    if duplicates:
                        st.warning(f"Skipping duplicate players: {', '.join(duplicates)}")
                    
                    if unique_new_players:
                        try:
                            # Add new players to the list
                            updated_player_names = player_names + unique_new_players
                            
                            # Create dataframe for updating
                            updated_players_df = pd.DataFrame({"player_names": updated_player_names})
                            
                            # Update the worksheet
                            conn.update(worksheet=player_names_worksheet, data=updated_players_df)
                            
                            # Clear cache and show success
                            st.cache_data.clear()
                            st.success(f"✅ Added {len(unique_new_players)} new players successfully!")
                            st.info("🔄 Page will refresh to show the updated player list.")
                            st.rerun()
                            
                        except Exception as e:
                            if "Worksheet not found" in str(e) or "does not exist" in str(e):
                                try:
                                    # Create worksheet with all players
                                    initial_players_df = pd.DataFrame({"player_names": unique_new_players})
                                    conn.update(worksheet=player_names_worksheet, data=initial_players_df)
                                    st.cache_data.clear()
                                    st.success(f"✅ Created worksheet '{player_names_worksheet}' and added {len(unique_new_players)} players!")
                                    st.info("🔄 Page will refresh to show the updated player list.")
                                    st.rerun()
                                except Exception as create_error:
                                    st.error(f"Error creating worksheet: {str(create_error)}")
                            else:
                                st.error(f"Error adding players: {str(e)}")
                    else:
                        st.info("No new players to add (all were duplicates).")
                else:
                    st.error("No valid player names found.")
            else:
                st.error("Please enter at least one player name.")
