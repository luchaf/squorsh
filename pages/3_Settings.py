import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
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

# Session Management
st.header("🎾 Session Management")
st.markdown("Create and manage different sessions (tournaments, seasons, events)")

# Function to refresh session data after changes
def refresh_session_data():
    """Refresh session data from Google Sheets after making changes"""
    try:
        # Clear all Streamlit caches to force fresh data
        st.cache_data.clear()
        if hasattr(st.cache_resource, 'clear'):
            st.cache_resource.clear()
        
        # Force fresh read with explicit parameters
        sessions_df = conn.read(
            worksheet="sessions", 
            usecols=None, 
            nrows=None,
            ttl=0  # No caching
        )
        
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
            # Force fresh read by clearing any potential cache
            st.cache_data.clear()
            
            # Read FRESH sessions data with explicit parameters
            fresh_sessions_df = conn.read(worksheet="sessions", usecols=None, nrows=None)
            
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
                    
                    # Refresh session data after deletion
                    sessions_df, existing_sessions = refresh_session_data()
                    
                    # Immediately rerun to refresh UI lists
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error deleting session: {str(e)}")
            else:
                st.error("❌ Confirmation text doesn't match!")

# Footer
st.divider()
st.markdown("---")
st.markdown("*Settings page - Manage your squash sessions with ease!*")
