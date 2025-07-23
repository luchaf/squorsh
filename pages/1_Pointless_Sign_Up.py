import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from color_palette import PRIMARY, SECONDARY, TERTIARY

st.set_page_config(page_title="Pointless Sign Up", layout="wide", page_icon="📝")

# Initialize connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Page header
st.title("📝 Pointless Sign Up")
st.markdown("*Register as a player and view match history*")

# Load current session
def get_current_session():
    """Get the current active session from Google Sheets"""
    try:
        sessions_df = conn.read(worksheet="sessions", ttl=0)
        if not sessions_df.empty and "active" in sessions_df.columns:
            active_sessions = sessions_df[sessions_df["active"] == True]
            if not active_sessions.empty:
                return active_sessions.iloc[0]["session_name"]
    except Exception:
        pass
    
    # Fallback: try to get first available session
    try:
        if not sessions_df.empty and "session_name" in sessions_df.columns:
            return sessions_df.iloc[0]["session_name"]
    except Exception:
        pass
    
    return None

current_session = get_current_session()

if not current_session:
    st.error("⚠️ No active session found. Please contact an administrator to set up a session.")
    st.info("💡 Sessions can be managed in the Settings page by administrators.")
    st.stop()

# Display current session
st.success(f"✅ Current Session: **{current_session}**")

# Create tabs
tab1, tab2 = st.tabs(["👤 Player Registration", "👥 Current Players"])

with tab1:
    st.header("👤 Player Registration")
    st.markdown("*Add yourself to the player list for the current session*")
    
    # Define worksheet names
    player_names_worksheet = f"{current_session}_player_names"
    
    # Load current players function (for registration logic)
    def load_players():
        try:
            player_names_df = conn.read(worksheet=player_names_worksheet, ttl=0)
            if not player_names_df.empty and "player_names" in player_names_df.columns:
                return [name for name in player_names_df["player_names"].tolist() if name and str(name).strip()]
            else:
                return []
        except Exception:
            return []
    
    with st.form("player_registration_form"):
        new_player_name = st.text_input(
            "Your Name",
            placeholder="Enter your full name...",
            help="Enter your full name to register for this session"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            register_button = st.form_submit_button("Register Me!", type="primary")
        
        if register_button:
            if new_player_name.strip():
                # Load current players for duplicate checking
                player_names = load_players()
                # Check if player already exists
                if new_player_name.strip() in player_names:
                    st.warning(f"⚠️ The name '{new_player_name}' is already registered!")
                    st.info("💡 Please choose a different name or check the 'Current Players' tab to see all registered names.")
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
                        st.success(f"🎉 Welcome '{new_player_name}'! You've been registered successfully!")
                        st.info("🔄 Page will refresh to show the updated player list.")
                        st.rerun()
                        
                    except Exception as e:
                        if "Worksheet not found" in str(e) or "does not exist" in str(e):
                            st.info(f"Creating new player list for session '{current_session}'...")
                            try:
                                # Try to create the worksheet by updating with initial data
                                initial_players_df = pd.DataFrame({"player_names": [new_player_name.strip()]})
                                conn.update(worksheet=player_names_worksheet, data=initial_players_df)
                                st.cache_data.clear()
                                st.success(f"🎉 Welcome '{new_player_name}'! You're the first player registered!")
                                st.info("🔄 Page will refresh to show the updated player list.")
                                st.rerun()
                            except Exception as create_error:
                                st.error(f"Error creating player list: {str(create_error)}")
                        else:
                            st.error(f"Error registering player: {str(e)}")
            else:
                st.error("Please enter a valid name.")
    
    # Instructions
    st.divider()
    st.subheader("ℹ️ How it works")
    st.markdown("""
    1. **Enter your name** in the form above
    2. **Click 'Register Me!'** to add yourself to the player list
    3. **You're ready to play!** Your name will appear in match entry forms
    4. **Start recording matches** in the 'Pointless Racquet Records' page
    """)

with tab2:
    st.header("👥 Current Players")
    st.markdown(f"*All registered players for session: **{current_session}***")
    
    # Load current players (same logic as in tab1)
    player_names = load_players()
    
    if player_names:
        # Display player count
        st.metric("Total Registered Players", len(player_names))
        
        st.divider()
        
        # Display players in a nice format
        st.subheader("Player List")
        
        # Create a nice display of current players
        players_df = pd.DataFrame({"Player Name": player_names})
        players_df.index = players_df.index + 1  # Start numbering from 1
        st.dataframe(players_df, use_container_width=True)
        
        # Optional: Show players in a more visual way
        st.divider()
        st.subheader("🎯 Player Grid")
        
        # Display players in columns for better visual layout
        num_cols = min(3, len(player_names))  # Max 3 columns
        if num_cols > 0:
            cols = st.columns(num_cols)
            for i, player in enumerate(player_names):
                with cols[i % num_cols]:
                    st.info(f"🎾 {player}")
    else:
        st.info("📧 No players registered yet for this session.")
        st.markdown("**Get started:**")
        st.markdown("1. Go to the 'Player Registration' tab above to register yourself")
        st.markdown("2. Invite other players to register themselves")
        st.markdown("3. Start recording matches in 'Pointless Racquet Records'")

# Footer
st.divider()
st.markdown("---")
st.markdown("*Pointless Sign Up - Join the game and track your progress!*")
