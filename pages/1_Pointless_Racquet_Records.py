import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import date

# Create GSheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

worksheet_name = "Sheet1"
df_sheet = conn.read(worksheet=worksheet_name)

(
    show_me_the_list,
    online_form,
) = st.tabs(["List of recorded matches", "Online form"])

with show_me_the_list:
    st.dataframe(df_sheet)
    df = df_sheet.copy()

    # Expander for Inserting Rows
    with st.expander("Insert Row"):
        insert_index = st.text_input('Enter index to insert the row (e.g., "5"):')
        new_row = {col: st.text_input(f"Enter value for {col}:") for col in df.columns}

        if st.button("Insert Row"):
            try:
                index_to_insert = int(insert_index)
                if index_to_insert < 0:
                    st.error("Invalid index. Please enter a non-negative index.")
                else:
                    upper_half = df.iloc[:index_to_insert]
                    lower_half = df.iloc[index_to_insert:]
                    new_df = pd.concat([upper_half, pd.DataFrame([new_row]), lower_half], ignore_index=True)
                    df = new_df
                    st.dataframe(df)
                    # Update worksheet
                    conn.update(worksheet=worksheet_name, data=df)
                    st.cache_data.clear()
                    st.success("Row inserted and data updated in Google Sheet.")
            except ValueError:
                st.error("Please enter a valid numeric index.")

    # Expander for Deleting Rows
    with st.expander("Delete Row"):
        delete_index = st.text_input('Enter row index to delete:')
        if st.button("Delete Row"):
            try:
                index_to_delete = int(delete_index)
                if 0 <= index_to_delete < len(df):
                    df = df.drop(index_to_delete).reset_index(drop=True)
                    st.dataframe(df)
                    # Update worksheet
                    conn.update(worksheet=worksheet_name, data=df)
                    st.cache_data.clear()
                    st.success("Row deleted and data updated in Google Sheet.")
                else:
                    st.error("Invalid index. Please enter a valid row index.")
            except ValueError:
                st.error("Please enter a valid numeric index.")

    # Expander for Updating Values
    with st.expander("Update Values"):
        selected_row = st.selectbox("Select a row to update:", range(len(df)))
        column_to_update = st.selectbox("Select a column to update:", df.columns)
        new_value = st.text_input(f"Enter a new value for {column_to_update}:")
        if st.button("Update Value"):
            try:
                df.at[selected_row, column_to_update] = new_value
                st.dataframe(df)
                # Update worksheet
                conn.update(worksheet=worksheet_name, data=df)
                st.cache_data.clear()
                st.success("Value updated and data saved to Google Sheet.")
            except Exception as e:
                st.error(f"Error updating value: {str(e)}")

with online_form:
    player_names = ["Friedemann", "Lucas", "Peter", "Simon", "Tobias"]

    def reset_session_state():
        """Helper function to reset session state."""
        keys = ['player1_name', 'player1_score', 'player2_name', 'player2_score', 'matchday_input', 'data_written']
        for key in keys:
            st.session_state[key] = None

def display_enter_match_results(df):
    # Initialize session state variables if not already set
    if 'data_written' not in st.session_state:
        st.session_state['data_written'] = False
    if 'player1_name' not in st.session_state:
        st.session_state['player1_name'] = player_names[0]  # Default to the first player
    if 'player1_score' not in st.session_state:
        st.session_state['player1_score'] = 0
    if 'player2_name' not in st.session_state:
        st.session_state['player2_name'] = player_names[0]  # Default to the first player
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
        st.session_state['player1_name'] = st.selectbox(
            "Player 1",
            player_names,
            index=player_names.index(st.session_state['player1_name']),
            key="player1_name"
        )
        st.session_state['player1_score'] = st.number_input(
            "Player 1 Score",
            min_value=0,
            step=1,
            value=st.session_state['player1_score'],
            key="player1_score"
        )
        st.session_state['player2_name'] = st.selectbox(
            "Player 2",
            player_names,
            index=player_names.index(st.session_state['player2_name']),
            key="player2_name"
        )
        st.session_state['player2_score'] = st.number_input(
            "Player 2 Score",
            min_value=0,
            step=1,
            value=st.session_state['player2_score'],
            key="player2_score"
        )
        st.session_state['matchday_input'] = st.date_input(
            "Matchday",
            value=st.session_state['matchday_input'],
            key="matchday_input"
        )

        # Submit button to save the match result
        if st.button("Submit Match Result"):
            new_data = {
                "Player1": st.session_state['player1_name'],
                "Score1": st.session_state['player1_score'],
                "Player2": st.session_state['player2_name'],
                "Score2": st.session_state['player2_score'],
                "date": st.session_state['matchday_input'].strftime('%Y-%m-%d'),
            }
            updated_df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            # Update worksheet
            conn.update(worksheet=worksheet_name, data=updated_df)
            st.cache_data.clear()
            st.session_state['data_written'] = True
            st.rerun()  # Rerun to show the success message

display_enter_match_results(df)
