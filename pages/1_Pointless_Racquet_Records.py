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
    st.header("🏓 Match Results Management")
    st.markdown(f"**Current Mode:** {mode}")
    st.markdown(f"**Managing matches for:** `{worksheet_name}`")
    
    # Display current matches
    st.subheader("Current Matches")
    if not df.empty:
        col1, col2 = st.columns([4, 1])
        with col1:
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
        with col2:
            st.metric("Total Matches", len(df))
            if not df.empty:
                latest_date = pd.to_datetime(df["date"], format="%Y%m%d").max()
                st.metric("Latest Match", latest_date.strftime("%Y-%m-%d"))
    else:
        st.info("No matches found. Add some matches in the 'Online form' tab!")
    
    # Only show edit/delete options if there are matches
    if not df.empty:
        st.divider()
        
        # Edit match section
        st.subheader("Edit Match Result")
        with st.form("edit_match_form"):
            # Select match to edit
            match_options = []
            for _, row in df.iterrows():
                match_date = pd.to_datetime(row["date"], format="%Y%m%d").strftime("%Y-%m-%d")
                match_desc = f"Match #{row['match_number_total']} ({match_date}): {row['Player1']} ({row['Score1']}) vs {row['Player2']} ({row['Score2']})"
                match_options.append((row['match_number_total'], match_desc))
            
            selected_match = st.selectbox(
                "Select match to edit",
                options=[opt[0] for opt in match_options],
                format_func=lambda x: next(opt[1] for opt in match_options if opt[0] == x),
                help="Choose a match to edit"
            )
            
            if selected_match:
                # Get the selected match data
                match_row = df[df["match_number_total"] == selected_match].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    edit_player1 = st.selectbox(
                        "Player 1",
                        options=player_names,
                        index=player_names.index(match_row["Player1"]) if match_row["Player1"] in player_names else 0,
                        key="edit_player1"
                    )
                    edit_score1 = st.number_input(
                        "Player 1 Score",
                        min_value=0,
                        value=int(match_row["Score1"]),
                        key="edit_score1"
                    )
                with col2:
                    edit_player2 = st.selectbox(
                        "Player 2",
                        options=player_names,
                        index=player_names.index(match_row["Player2"]) if match_row["Player2"] in player_names else 0,
                        key="edit_player2"
                    )
                    edit_score2 = st.number_input(
                        "Player 2 Score",
                        min_value=0,
                        value=int(match_row["Score2"]),
                        key="edit_score2"
                    )
                
                edit_date = st.date_input(
                    "Match Date",
                    value=pd.to_datetime(match_row["date"], format="%Y%m%d").date(),
                    key="edit_date"
                )
                
                edit_password = st.text_input(
                    "Admin Password",
                    type="password",
                    help="Enter the admin password to confirm edit",
                    key="edit_password"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    edit_button = st.form_submit_button("Update Match", type="primary")
                
                if edit_button:
                    if not edit_password:
                        st.error("Password is required to edit matches.")
                    else:
                        try:
                            admin_password = st.secrets["admin_password"]
                            if edit_password != admin_password:
                                st.error("Incorrect password. Match edit denied.")
                            else:
                                try:
                                    # Update the match data
                                    updated_df = df.copy()
                                    match_index = updated_df[updated_df["match_number_total"] == selected_match].index[0]
                                    
                                    # Update the row
                                    updated_df.loc[match_index, "Player1"] = edit_player1
                                    updated_df.loc[match_index, "Score1"] = edit_score1
                                    updated_df.loc[match_index, "Player2"] = edit_player2
                                    updated_df.loc[match_index, "Score2"] = edit_score2
                                    updated_df.loc[match_index, "date"] = edit_date.strftime("%Y%m%d")
                                    
                                    # Recalculate match_number_day for the new date
                                    new_date_str = edit_date.strftime("%Y%m%d")
                                    day_matches = updated_df[updated_df["date"] == new_date_str]
                                    if len(day_matches) > 1:
                                        # There are other matches on this day, assign next number
                                        other_day_matches = day_matches[day_matches["match_number_total"] != selected_match]
                                        if not other_day_matches.empty:
                                            updated_df.loc[match_index, "match_number_day"] = other_day_matches["match_number_day"].max() + 1
                                        else:
                                            updated_df.loc[match_index, "match_number_day"] = 1
                                    else:
                                        # This is the only match on this day
                                        updated_df.loc[match_index, "match_number_day"] = 1
                                    
                                    # Update worksheet
                                    conn.update(worksheet=worksheet_name, data=updated_df)
                                    st.cache_data.clear()
                                    st.success(f"✅ Match #{selected_match} updated successfully!")
                                    st.info("🔄 Page will refresh to show the updated match list.")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Error updating match: {str(e)}")
                        except KeyError:
                            st.error("Admin password not found in secrets. Please configure 'admin_password' in your Streamlit secrets.")
        
        st.divider()
        
        # Delete match section
        st.subheader("Delete Match Result")
        
        # Single match deletion
        st.markdown("**Remove Single Match**")
        with st.form("delete_match_form"):
            match_to_delete = st.selectbox(
                "Select match to delete",
                options=[opt[0] for opt in match_options],
                format_func=lambda x: next(opt[1] for opt in match_options if opt[0] == x),
                help="Choose a match to remove"
            )
            
            delete_password = st.text_input(
                "Admin Password",
                type="password",
                help="Enter the admin password to confirm deletion"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                delete_button = st.form_submit_button("Delete Match", type="secondary")
            
            if delete_button:
                if not delete_password:
                    st.error("Password is required to delete matches.")
                elif not match_to_delete:
                    st.error("Please select a match to delete.")
                else:
                    try:
                        admin_password = st.secrets["admin_password"]
                        if delete_password != admin_password:
                            st.error("Incorrect password. Match deletion denied.")
                        else:
                            try:
                                # Remove match from the dataframe
                                updated_df = df[df["match_number_total"] != match_to_delete].copy()
                                
                                # Update worksheet
                                conn.update(worksheet=worksheet_name, data=updated_df)
                                st.cache_data.clear()
                                st.success(f"✅ Match #{match_to_delete} deleted successfully!")
                                st.info("🔄 Page will refresh to show the updated match list.")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error deleting match: {str(e)}")
                    except KeyError:
                        st.error("Admin password not found in secrets. Please configure 'admin_password' in your Streamlit secrets.")
        
        # Bulk match deletion
        st.markdown("**Remove Multiple Matches**")
        with st.form("bulk_delete_matches_form"):
            matches_to_delete = st.multiselect(
                "Select matches to delete",
                options=[opt[0] for opt in match_options],
                format_func=lambda x: next(opt[1] for opt in match_options if opt[0] == x),
                help="Choose multiple matches to remove"
            )
            
            bulk_delete_password = st.text_input(
                "Admin Password",
                type="password",
                help="Enter the admin password to confirm deletion",
                key="bulk_delete_match_password"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                bulk_delete_button = st.form_submit_button("Delete Selected", type="secondary")
            
            if bulk_delete_button:
                if not bulk_delete_password:
                    st.error("Password is required to delete matches.")
                elif not matches_to_delete:
                    st.error("Please select at least one match to delete.")
                else:
                    try:
                        admin_password = st.secrets["admin_password"]
                        if bulk_delete_password != admin_password:
                            st.error("Incorrect password. Match deletion denied.")
                        else:
                            try:
                                # Remove selected matches from the dataframe
                                updated_df = df[~df["match_number_total"].isin(matches_to_delete)].copy()
                                
                                # Update worksheet
                                conn.update(worksheet=worksheet_name, data=updated_df)
                                st.cache_data.clear()
                                st.success(f"✅ Deleted {len(matches_to_delete)} matches successfully!")
                                st.info("🔄 Page will refresh to show the updated match list.")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error deleting matches: {str(e)}")
                    except KeyError:
                        st.error("Admin password not found in secrets. Please configure 'admin_password' in your Streamlit secrets.")

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
    
    # Delete players section
    if player_names:  # Only show delete options if there are players
        st.divider()
        st.subheader("Delete Players")
        
        # Single player deletion
        st.markdown("**Remove Single Player**")
        with st.form("delete_player_form"):
            player_to_delete = st.selectbox(
                "Select player to delete",
                options=player_names,
                help="Choose a player to remove from the list"
            )
            
            delete_password = st.text_input(
                "Admin Password",
                type="password",
                help="Enter the admin password to confirm deletion"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                delete_button = st.form_submit_button("Delete Player", type="secondary")
            
            if delete_button:
                if not delete_password:
                    st.error("Password is required to delete players.")
                elif not player_to_delete:
                    st.error("Please select a player to delete.")
                else:
                    try:
                        admin_password = st.secrets["admin_password"]
                        if delete_password != admin_password:
                            st.error("Incorrect password. Player deletion denied.")
                        else:
                            # Password is correct, proceed with deletion
                            try:
                                # Remove player from the list
                                updated_player_names = [p for p in player_names if p != player_to_delete]
                                
                                # Create dataframe for updating
                                updated_players_df = pd.DataFrame({"player_names": updated_player_names})
                                
                                # Update the worksheet
                                conn.update(worksheet=player_names_worksheet, data=updated_players_df)
                                
                                # Clear cache and show success
                                st.cache_data.clear()
                                st.success(f"✅ Player '{player_to_delete}' deleted successfully!")
                                st.info("🔄 Page will refresh to show the updated player list.")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error deleting player: {str(e)}")
                    except KeyError:
                        st.error("Admin password not found in secrets. Please configure 'admin_password' in your Streamlit secrets.")
        
        # Bulk player deletion
        st.markdown("**Remove Multiple Players**")
        with st.form("bulk_delete_players_form"):
            players_to_delete = st.multiselect(
                "Select players to delete",
                options=player_names,
                help="Choose multiple players to remove from the list"
            )
            
            bulk_delete_password = st.text_input(
                "Admin Password",
                type="password",
                help="Enter the admin password to confirm deletion",
                key="bulk_delete_password"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                bulk_delete_button = st.form_submit_button("Delete Selected", type="secondary")
            
            if bulk_delete_button:
                if not bulk_delete_password:
                    st.error("Password is required to delete players.")
                elif not players_to_delete:
                    st.error("Please select at least one player to delete.")
                else:
                    try:
                        admin_password = st.secrets["admin_password"]
                        if bulk_delete_password != admin_password:
                            st.error("Incorrect password. Player deletion denied.")
                        else:
                            # Password is correct, proceed with deletion
                            try:
                                # Remove selected players from the list
                                updated_player_names = [p for p in player_names if p not in players_to_delete]
                                
                                # Create dataframe for updating
                                updated_players_df = pd.DataFrame({"player_names": updated_player_names})
                                
                                # Update the worksheet
                                conn.update(worksheet=player_names_worksheet, data=updated_players_df)
                                
                                # Clear cache and show success
                                st.cache_data.clear()
                                st.success(f"✅ Deleted {len(players_to_delete)} players successfully!")
                                st.info("🔄 Page will refresh to show the updated player list.")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error deleting players: {str(e)}")
                    except KeyError:
                        st.error("Admin password not found in secrets. Please configure 'admin_password' in your Streamlit secrets.")
