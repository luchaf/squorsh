import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from color_palette import PRIMARY, SECONDARY, TERTIARY

st.set_page_config(page_title="Settings", layout="wide", page_icon="⚙️")

# Create GSheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Get admin password from secrets
try:
    admin_password = st.secrets["admin_password"]
except KeyError:
    st.error("❌ Admin password not found in secrets.toml!")
    st.info("Please add 'admin_password = \"your_password\"' to your secrets.toml file")
    st.stop()

st.title("⚙️ Settings")
st.markdown("*Session Management & Configuration*")

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("🔐 Admin Authentication Required")
    
    with st.form("auth_form"):
        password_input = st.text_input(
            "Admin Password",
            type="password",
            help="Enter the admin password to access settings"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            login_button = st.form_submit_button("Login", type="primary")
        
        if login_button:
            if password_input == admin_password:
                st.session_state.authenticated = True
                st.success("✅ Authentication successful!")
                st.rerun()
            else:
                st.error("❌ Invalid password!")
    
    st.stop()

# Main settings interface (only shown when authenticated)
st.success("🔓 Admin access granted")

# Logout button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🚪 Logout", type="secondary"):
        st.session_state.authenticated = False
        st.rerun()

st.divider()

# Main Settings Tabs
tab1, tab2, tab3 = st.tabs(["🎾 Session Management", "👥 Player Management", "🏓 Match Management"])

with tab1:
    # Session Management
    st.header("🎾 Session Management")
    st.markdown("Create and manage different sessions (tournaments, seasons, events)")

    # Function to refresh session data after changes
    @st.cache_data(ttl=30)  # Cache for 30 seconds
    def get_cached_session_data():
        """Get session data with caching to reduce API calls"""
        return conn.read(worksheet="sessions", usecols=None, nrows=None)

    def refresh_session_data():
        """Refresh session data from Google Sheets after making changes"""
        try:
            # Only clear cache when absolutely necessary
            if st.session_state.get('force_refresh_sessions', False):
                st.cache_data.clear()
                st.session_state.force_refresh_sessions = False
            
            # Use cached read
            sessions_df = get_cached_session_data()
            
            if sessions_df.empty or "session_name" not in sessions_df.columns:
                return pd.DataFrame(columns=["session_name", "active"]), []
            else:
                # Ensure active column exists
                if "active" not in sessions_df.columns:
                    sessions_df["active"] = False
                # Convert to proper boolean type
                sessions_df["active"] = sessions_df["active"].astype(bool)
                existing_sessions = sessions_df["session_name"].tolist()
                return sessions_df, existing_sessions
        except Exception as e:
            st.error(f"Error refreshing session data: {str(e)}")
            return pd.DataFrame(columns=["session_name", "active"]), []

    # Load initial session data using the refresh function for consistency
    sessions_df, existing_sessions = refresh_session_data()

    # Display stored messages from previous actions
    if "delete_success" in st.session_state:
        st.success(st.session_state.delete_success)
        del st.session_state.delete_success

    if "delete_worksheets_success" in st.session_state:
        st.success(st.session_state.delete_worksheets_success)
        del st.session_state.delete_worksheets_success

    if "delete_worksheets_warning" in st.session_state:
        st.warning(st.session_state.delete_worksheets_warning)
        del st.session_state.delete_worksheets_warning

    if "delete_worksheets_info" in st.session_state:
        st.info(st.session_state.delete_worksheets_info)
        del st.session_state.delete_worksheets_info

    # Debug info (remove in production)
    if st.checkbox("🔍 Show Debug Info", value=False):
        st.write(f"**Sessions loaded:** {len(existing_sessions)}")
        st.write(f"**Session names:** {existing_sessions}")
        if not sessions_df.empty:
            st.write(f"**Active sessions:** {sessions_df[sessions_df['active'] == True]['session_name'].tolist()}")
        st.write(f"**DataFrame shape:** {sessions_df.shape}")

    # Display existing sessions
    st.subheader("📋 Existing Sessions")
    if existing_sessions:
        # Create display dataframe with active status
        display_data = []
        for _, row in sessions_df.iterrows():
            session_name = row["session_name"]
            is_active = row.get("active", False)
            status = "✅ Active" if is_active else "⏸️ Inactive"
            display_data.append({"Session Name": session_name, "Status": status})
        
        sessions_display_df = pd.DataFrame(display_data)
        sessions_display_df.index = sessions_display_df.index + 1
        st.dataframe(sessions_display_df, use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Total Sessions", len(existing_sessions))
        with col2:
            # Show centrally active session from Google Sheets
            active_sessions = sessions_df[sessions_df["active"] == True]
            central_active = active_sessions.iloc[0]["session_name"] if not active_sessions.empty else "None"
            st.metric("Central Active Session", central_active)
    else:
        st.info("No sessions found. Create your first session below!")

    st.divider()

    # Create new session
    st.subheader("➕ Create New Session")

    with st.form("create_session_form"):
        new_session_name = st.text_input(
            "Session Name",
            placeholder="e.g., 'Winter_Tournament_2024', 'League_Season_1', 'Friday_Games'",
            help="Use descriptive names. Avoid spaces and special characters."
        )
        
        session_description = st.text_area(
            "Description (Optional)",
            placeholder="Brief description of this session...",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            create_button = st.form_submit_button("Create Session", type="primary")
        
        if create_button:
            if new_session_name.strip():
                # Clean session name (remove spaces, special chars)
                clean_session_name = new_session_name.strip().replace(" ", "_")
                clean_session_name = "".join(c for c in clean_session_name if c.isalnum() or c in "_-")
                
                if clean_session_name in existing_sessions:
                    st.error(f"❌ Session '{clean_session_name}' already exists!")
                else:
                    try:
                        # Create the two worksheets for this session
                        player_names_worksheet = f"{clean_session_name}_player_names"
                        match_results_worksheet = f"{clean_session_name}_match_results"
                        
                        st.info(f"🔄 Creating session '{clean_session_name}'...")
                        
                        # Create initial dataframes with at least one row to ensure worksheet creation
                        initial_players_df = pd.DataFrame({"player_names": [""]})
                        initial_matches_df = pd.DataFrame({
                            "Player1": [""],
                            "Player2": [""],
                            "Score1": [""],
                            "Score2": [""],
                            "date": [""],
                            "match_number_total": [""],
                            "match_number_day": [""]
                        })
                        
                        # Create the worksheets using conn.create()
                        st.info(f"📋 Creating worksheet: {player_names_worksheet}")
                        try:
                            conn.create(worksheet=player_names_worksheet, data=initial_players_df)
                            st.success(f"✅ Created {player_names_worksheet}")
                        except Exception as e:
                            st.error(f"❌ Error creating {player_names_worksheet}: {str(e)}")
                            raise e
                        
                        st.info(f"📋 Creating worksheet: {match_results_worksheet}")
                        try:
                            conn.create(worksheet=match_results_worksheet, data=initial_matches_df)
                            st.success(f"✅ Created {match_results_worksheet}")
                        except Exception as e:
                            st.error(f"❌ Error creating {match_results_worksheet}: {str(e)}")
                            raise e
                        
                        # Add to sessions list with active status
                        try:
                            sessions_df = conn.read(worksheet="sessions")
                            if sessions_df.empty or "session_name" not in sessions_df.columns:
                                # Create new sessions worksheet with active column
                                sessions_df = pd.DataFrame({
                                    "session_name": [clean_session_name],
                                    "active": [True]  # First session is automatically active
                                })
                            else:
                                # Add active column if it doesn't exist
                                if "active" not in sessions_df.columns:
                                    sessions_df["active"] = False
                                
                                # Set all existing sessions to inactive
                                sessions_df["active"] = False
                                
                                # Add new session as active
                                new_session = pd.DataFrame({
                                    "session_name": [clean_session_name],
                                    "active": [True]
                                })
                                sessions_df = pd.concat([sessions_df, new_session], ignore_index=True)
                        except Exception:
                            # Create new sessions worksheet
                            sessions_df = pd.DataFrame({
                                "session_name": [clean_session_name],
                                "active": [True]
                            })
                        
                        st.info("📋 Updating sessions list...")
                        try:
                            conn.update(worksheet="sessions", data=sessions_df)
                            st.success("✅ Updated sessions list")
                        except Exception as sessions_error:
                            st.warning(f"Sessions worksheet doesn't exist, creating it...")
                            try:
                                conn.create(worksheet="sessions", data=sessions_df)
                                st.success("✅ Created sessions worksheet")
                            except Exception as create_error:
                                st.error(f"❌ Error creating sessions worksheet: {str(create_error)}")
                                # Continue anyway - session worksheets were created
                        
                        st.success(f"✅ Session '{clean_session_name}' created successfully!")
                        st.info(f"📊 Created worksheets: '{player_names_worksheet}' and '{match_results_worksheet}'")
                        
                        # Set as current session
                        st.session_state.current_session = clean_session_name
                        
                        # Clear cache to show new session immediately
                        st.cache_data.clear()
                        
                        # Refresh session data after creation
                        sessions_df, existing_sessions = refresh_session_data()
                        
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error creating session: {str(e)}")
                        st.error("Please check:")
                        st.error("1. Google Sheet is shared with your Service Account email")
                        st.error("2. Service Account has 'Editor' permissions")
                        st.error("3. Google Sheets API is enabled in your Google Cloud project")
            else:
                st.error("❌ Please enter a valid session name!")

    # Session selection
    if existing_sessions:
        st.divider()
        st.subheader("🎯 Select Active Session")
        
        current_session = st.session_state.get("current_session", existing_sessions[0] if existing_sessions else "")
        
        selected_session = st.selectbox(
            "Choose the active session for recording matches and managing players:",
            options=existing_sessions,
            index=existing_sessions.index(current_session) if current_session in existing_sessions else 0,
            help="This session will be used in the 'Pointless Racquet Records' page"
        )
        
        if st.button("🎾 Set Active Session", type="primary"):
            try:
                # Use cached session data to avoid rate limits
                fresh_sessions_df = get_cached_session_data()
                
                # Use the fresh data as the source of truth
                actual_sessions = fresh_sessions_df["session_name"].tolist() if "session_name" in fresh_sessions_df.columns else []
                
                # Verify that the selected session exists in the fresh data
                if selected_session not in actual_sessions:
                    st.error(f"❌ ERROR: Selected session '{selected_session}' not found in current data!")
                    st.error(f"Available sessions: {actual_sessions}")
                    st.stop()
                
                # Add active column if it doesn't exist
                if "active" not in fresh_sessions_df.columns:
                    fresh_sessions_df["active"] = False
                
                # Set all sessions to inactive
                fresh_sessions_df["active"] = False
                
                # Set selected session to active
                fresh_sessions_df.loc[fresh_sessions_df["session_name"] == selected_session, "active"] = True
                
                # Update the sessions worksheet
                conn.update(worksheet="sessions", data=fresh_sessions_df)
                
                # Also update session state for immediate effect
                st.session_state.current_session = selected_session
                
                # Clear cache to show changes immediately
                st.cache_data.clear()
                
                # Refresh session data after update
                sessions_df, existing_sessions = refresh_session_data()
                
                st.success(f"✅ Active session set to: **{selected_session}** (for all devices)")
                st.info("🔄 This change will apply to all devices accessing the app!")
                st.rerun()  # Refresh to show updated data
                
            except Exception as e:
                st.error(f"❌ Error setting active session: {str(e)}")
                st.exception(e)

    # Session deletion (advanced)
    if existing_sessions:
        st.divider()
        st.subheader("🗑️ Delete Session")
        st.warning("⚠️ **Danger Zone**: Deleting a session will remove all associated data!")
        
        with st.expander("Delete Session (Advanced)", expanded=False):
            session_to_delete = st.selectbox(
                "Select session to delete:",
                options=existing_sessions,
                help="This will permanently delete the session and all its data"
            )
            
            confirm_delete = st.text_input(
                f"Type '{session_to_delete}' to confirm deletion:",
                help="This is a safety measure to prevent accidental deletion"
            )
            
            if st.button("🗑️ Delete Session", type="secondary"):
                if confirm_delete == session_to_delete:
                    try:
                        # First, try to delete the associated worksheets using gspread directly
                        player_names_worksheet = f"{session_to_delete}_player_names"
                        match_results_worksheet = f"{session_to_delete}_match_results"
                        
                        worksheets_deleted = []
                        worksheets_failed = []
                        
                        # Use gspread directly with service account to delete worksheets
                        try:
                            import gspread
                            from google.oauth2.service_account import Credentials
                            
                            # Get service account info from Streamlit secrets
                            service_account_info = {
                                "type": "service_account",
                                "project_id": st.secrets["connections"]["gsheets"]["project_id"],
                                "private_key_id": st.secrets["connections"]["gsheets"]["private_key_id"],
                                "private_key": st.secrets["connections"]["gsheets"]["private_key"],
                                "client_email": st.secrets["connections"]["gsheets"]["client_email"],
                                "client_id": st.secrets["connections"]["gsheets"]["client_id"],
                                "auth_uri": st.secrets["connections"]["gsheets"]["auth_uri"],
                                "token_uri": st.secrets["connections"]["gsheets"]["token_uri"],
                                "auth_provider_x509_cert_url": st.secrets["connections"]["gsheets"]["auth_provider_x509_cert_url"],
                                "client_x509_cert_url": st.secrets["connections"]["gsheets"]["client_x509_cert_url"]
                            }
                            
                            # Create gspread client
                            credentials = Credentials.from_service_account_info(
                                service_account_info,
                                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
                            )
                            gc = gspread.authorize(credentials)
                            
                            # Open the spreadsheet
                            spreadsheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
                            spreadsheet = gc.open_by_url(spreadsheet_url)
                            
                            # Try to delete player names worksheet
                            try:
                                worksheet_obj = spreadsheet.worksheet(player_names_worksheet)
                                spreadsheet.del_worksheet(worksheet_obj)
                                worksheets_deleted.append(player_names_worksheet)
                            except Exception:
                                worksheets_failed.append(player_names_worksheet)
                            
                            # Try to delete match results worksheet
                            try:
                                worksheet_obj = spreadsheet.worksheet(match_results_worksheet)
                                spreadsheet.del_worksheet(worksheet_obj)
                                worksheets_deleted.append(match_results_worksheet)
                            except Exception:
                                worksheets_failed.append(match_results_worksheet)
                                
                        except Exception as e:
                            # Fallback: if we can't access gspread directly, add to failed list
                            st.error(f"Could not access gspread directly: {str(e)}")
                            worksheets_failed.extend([player_names_worksheet, match_results_worksheet])
                        
                        # Now remove from sessions list while preserving active column
                        current_sessions_df = conn.read(worksheet="sessions")
                        if not current_sessions_df.empty:
                            # Remove the session but keep all columns
                            updated_sessions_df = current_sessions_df[current_sessions_df["session_name"] != session_to_delete].copy()
                            conn.update(worksheet="sessions", data=updated_sessions_df)
                        else:
                            # Fallback if read fails
                            updated_sessions = [s for s in existing_sessions if s != session_to_delete]
                            sessions_df = pd.DataFrame({"session_name": updated_sessions, "active": [False] * len(updated_sessions)})
                            conn.update(worksheet="sessions", data=sessions_df)
                        
                        # Clear current session if it was deleted
                        if st.session_state.get("current_session") == session_to_delete:
                            st.session_state.current_session = updated_sessions[0] if updated_sessions else ""
                        
                        # Store success messages in session state
                        st.session_state.delete_success = f"✅ Session '{session_to_delete}' deleted successfully!"
                        if worksheets_deleted:
                            st.session_state.delete_worksheets_success = f"✅ Cleared worksheets: {', '.join(worksheets_deleted)}"
                        if worksheets_failed:
                            st.session_state.delete_worksheets_warning = f"⚠️ Could not clear worksheets: {', '.join(worksheets_failed)}"
                            st.session_state.delete_worksheets_info = "📝 These worksheets may need to be manually deleted from Google Sheets"
                        
                        # Clear cache to show changes immediately
                        st.cache_data.clear()
                        
                        # Refresh session data after deletion
                        sessions_df, existing_sessions = refresh_session_data()
                        
                        # Immediately rerun to refresh UI lists
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error deleting session: {str(e)}")
                else:
                    st.error("❌ Confirmation text doesn't match!")

with tab2:
    st.header("🎾 Player Management")
    
    # Load current session and player data
    current_session = st.session_state.get("current_session", "")
    
    if not current_session:
        # Try to get active session from Google Sheets
        try:
            fresh_sessions_df = conn.read(worksheet="sessions", ttl=0)
            if not fresh_sessions_df.empty and "active" in fresh_sessions_df.columns:
                active_sessions = fresh_sessions_df[fresh_sessions_df["active"] == True]
                if not active_sessions.empty:
                    current_session = active_sessions.iloc[0]["session_name"]
                    st.session_state["current_session"] = current_session
        except Exception:
            pass
    
    if not current_session:
        st.warning("⚠️ No active session found. Please create and activate a session in the Session Management tab first.")
        st.info("🔄 Go to the 'Session Management' tab to create or activate a session.")
    else:
        # Define worksheet names
        player_names_worksheet = f"{current_session}_player_names"
        
        # Load player names
        try:
            player_names_df = conn.read(worksheet=player_names_worksheet, ttl=0)
            if not player_names_df.empty and "player_names" in player_names_df.columns:
                player_names = [name for name in player_names_df["player_names"].tolist() if name and str(name).strip()]
            else:
                player_names = []
        except Exception:
            player_names = []
        
        st.markdown(f"**Current Session:** {current_session}")
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
                            
                            # Mark for refresh on next load
                            st.session_state.force_refresh_sessions = True
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
with tab3:
    # Match Management
    st.header("🏓 Match Management")
    st.markdown("Edit and delete match results")
    
    # Load current session and match data
    try:
        sessions_df, existing_sessions = refresh_session_data()
        
        if sessions_df.empty:
            st.warning("⚠️ No sessions found. Please create a session first in the Session Management tab.")
        else:
            # Get active session
            active_sessions = sessions_df[sessions_df["active"] == True]
            if not active_sessions.empty:
                current_session = active_sessions.iloc[0]["session_name"]
                st.info(f"✅ Managing matches for active session: **{current_session}**")
            else:
                current_session = existing_sessions[0] if existing_sessions else None
                st.info(f"🔄 No active session set. Using: **{current_session}**")
            
            if current_session:
                # Load match data for current session
                worksheet_name = f"{current_session}_match_results"
                
                try:
                    df = conn.read(worksheet=worksheet_name)
                    
                    if df.empty:
                        st.info("No matches found for this session.")
                    else:
                        st.success(f"Found {len(df)} matches in session '{current_session}'")
                        
                        # Load player names for dropdowns
                        player_names_worksheet = f"{current_session}_player_names"
                        try:
                            players_df = conn.read(worksheet=player_names_worksheet)
                            player_names = players_df["player_names"].tolist() if not players_df.empty and "player_names" in players_df.columns else []
                        except:
                            player_names = []
                        
                        # Only show edit/delete options if there are matches
                        if not df.empty:
                            st.divider()
                            
                            # Create match options for all forms
                            match_options = []
                            for _, row in df.iterrows():
                                match_date = pd.to_datetime(row["date"], format="%Y%m%d").strftime("%Y-%m-%d")
                                match_desc = f"Match #{row['match_number_total']} ({match_date}): {row['Player1']} ({row['Score1']}) vs {row['Player2']} ({row['Score2']})"
                                match_options.append((row['match_number_total'], match_desc))
                            
                            # Edit match section
                            st.subheader("Edit Match Result")
                            with st.form("edit_match_form"):
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
                                        if player_names:
                                            edit_player1 = st.selectbox(
                                                "Player 1",
                                                options=player_names,
                                                index=player_names.index(match_row["Player1"]) if match_row["Player1"] in player_names else 0,
                                                key="edit_player1"
                                            )
                                        else:
                                            edit_player1 = st.text_input(
                                                "Player 1",
                                                value=match_row["Player1"],
                                                key="edit_player1"
                                            )
                                        edit_score1 = st.number_input(
                                            "Player 1 Score",
                                            min_value=0,
                                            value=int(match_row["Score1"]),
                                            key="edit_score1"
                                        )
                                    with col2:
                                        if player_names:
                                            edit_player2 = st.selectbox(
                                                "Player 2",
                                                options=player_names,
                                                index=player_names.index(match_row["Player2"]) if match_row["Player2"] in player_names else 0,
                                                key="edit_player2"
                                            )
                                        else:
                                            edit_player2 = st.text_input(
                                                "Player 2",
                                                value=match_row["Player2"],
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
                            
                except Exception as e:
                    st.error(f"Error loading match data: {str(e)}")
                    st.info("Make sure the session has been created and has match data.")
            else:
                st.error("No current session available.")
    except Exception as e:
        st.error(f"Error loading session data: {str(e)}")

# Footer (outside all tabs)
st.divider()
st.markdown("---")
st.markdown("*Settings page - Manage your squash sessions and players with ease!*")
