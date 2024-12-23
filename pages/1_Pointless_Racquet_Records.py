import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import date

# Create GSheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

worksheet_name = "Sheet1"
df = conn.read(worksheet=worksheet_name)

(
    online_form,
    show_me_the_list,
) = st.tabs(["Online form", "List of recorded matches"])


with online_form:
    player_names = ["Friedemann", "Lucas", "Peter", "Simon", "Tobias"]


    def reset_session_state():
        """Helper function to reset session state."""
        st.session_state['player1_name'] = player_names[0]  # Default to the first player
        st.session_state['player1_score'] = 0
        st.session_state['player2_name'] = player_names[3]  # Default to the first player
        st.session_state['player2_score'] = 0
        st.session_state['matchday_input'] = date.today()
        st.session_state['data_written'] = False


    def display_enter_match_results(df):
        # Initialize session state variables if not already set
        if 'data_written' not in st.session_state:
            st.session_state['data_written'] = False
        if 'player1_name' not in st.session_state:
            st.session_state['player1_name'] = player_names[0]  # Default to the first player
        if 'player1_score' not in st.session_state:
            st.session_state['player1_score'] = 0
        if 'player2_name' not in st.session_state:
            st.session_state['player2_name'] = player_names[3]  # Default to the first player
        if 'player2_score' not in st.session_state:
            st.session_state['player2_score'] = 0
        if 'matchday_input' not in st.session_state:
            st.session_state['matchday_input'] = date.today()
    
        if st.session_state['data_written']:
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
                index=player_names.index(st.session_state['player1_name']),
                key="player1_name"
            )
            player1_score = st.number_input(
                "Player 1 Score",
                min_value=0,
                step=1,
                value=st.session_state['player1_score'],
                key="player1_score"
            )
            player2_name = st.selectbox(
                "Player 2",
                player_names,
                index=player_names.index(st.session_state['player2_name']),
                key="player2_name"
            )
            player2_score = st.number_input(
                "Player 2 Score",
                min_value=0,
                step=1,
                value=st.session_state['player2_score'],
                key="player2_score"
            )
            matchday_input = st.date_input(
                "Matchday",
                value=st.session_state['matchday_input'],
                key="matchday_input"
            )
    
            # Submit button to save the match result
            if st.button("Submit Match Result"):
                new_data = {
                    "Player1": player1_name,
                    "Score1": player1_score,
                    "Player2": player2_name,
                    "Score2": player2_score,
                    "date": matchday_input.strftime('%Y-%m-%d'),
                }
                updated_df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                # Update worksheet
                conn.update(worksheet=worksheet_name, data=updated_df)
                st.cache_data.clear()
                st.session_state['data_written'] = True
                st.rerun()  # Rerun to show the success message

    
    display_enter_match_results(df)

with show_me_the_list:
    st.dataframe(df)


