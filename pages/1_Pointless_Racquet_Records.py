import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from google.oauth2 import service_account
import tempfile
import os
from pointless_utils import extract_data_from_games
import pandas as pd
from datetime import date
from pathlib import Path

(
    show_me_the_list,
    online_form,
    upload_page,
    email,
    voice,
) = st.tabs([
    "List of recorded matches",
    "Online form",
    "Upload page",
    "Email",
    "Voice",
    ])

with show_me_the_list:
    # Load data from database
    df = pd.DataFrame()
    df_sheet = pd.read_csv(st.secrets["public_gsheets_url"])
    df_sheet["date"] = df_sheet["date"].astype(str)
    list_of_available_dates = list(set(df_sheet["date"].tolist()))
    df_sheet['parsed_sheet_df'] = df_sheet.apply(lambda x: extract_data_from_games(x["games"], x["date"]), axis=1)
    df_tmp = pd.DataFrame()
    for _, row in df_sheet.iterrows():
        df_tmp = pd.concat([df_tmp, row["parsed_sheet_df"]])
    df_tmp = df_tmp[["First Name", "First Score", "Second Name", "Second Score", "date"]].reset_index(drop=True).copy()
    df_tmp = df_tmp.rename(columns={
        "First Name": "Player1", 
        "Second Name": "Player2", 
        "First Score": "Score1", 
        "Second Score": "Score2", 
    }).copy()
    st.dataframe(df_tmp)

    df = df_tmp.copy()

    # Expander for Inserting Rows
    with st.expander("Insert Row"):
        insert_index = st.text_input('Enter index to insert the row (e.g., "5"):')
        new_row = {}
        for column in df.columns:
            new_value = st.text_input(f"Enter value for {column}:")
            new_row[column] = new_value

        if st.button("Insert"):
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
            except ValueError:
                st.error("Please enter a valid numeric index.")

    # Expander for Deleting Rows
    with st.expander("Delete Row"):
        delete_index = st.text_input('Enter row index to delete:')
        if st.button("Delete"):
            try:
                index_to_delete = int(delete_index)
                if index_to_delete >= 0 and index_to_delete < len(df):
                    df = df.drop(index_to_delete)
                    df.reset_index(drop=True, inplace=True)  # Reset the index
                    st.dataframe(df)
                else:
                    st.error("Invalid index. Please enter a valid row index.")
            except ValueError:
                st.error("Please enter a valid numeric index.")

    # Expander for Updating Values
    with st.expander("Update Values"):
        selected_row = st.selectbox("Select a row to update:", range(len(df)))
        column_to_update = st.selectbox("Select a column to update:", df.columns)
        new_value = st.text_input(f"Enter a new value for {column_to_update}:")
        if st.button("Update"):
            try:
                df.at[selected_row, column_to_update] = new_value
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error updating value: {str(e)}")


with online_form:
    # Define a list of player names
    player_names = ["Friedemann", "Lucas", "Peter", "Simon", "Tobias"]

    def reset_session_state():
        """Helper function to reset session state."""
        st.session_state['player1_name'] = ''
        st.session_state['player1_score'] = None
        st.session_state['player2_name'] = ''
        st.session_state['player2_score'] = None
        st.session_state['matchday_input'] = None
        st.session_state['show_confirm'] = False
        st.session_state['data_written'] = False
        
    def display_enter_match_results():
        # Initialize session state values if not already set
        if 'player1_name' not in st.session_state:
            st.session_state['player1_name'] = ''
        if 'player1_score' not in st.session_state:
            st.session_state['player1_score'] = None
        if 'player2_name' not in st.session_state:
            st.session_state['player2_name'] = ''
        if 'player2_score' not in st.session_state:
            st.session_state['player2_score'] = None
        if 'matchday_input' not in st.session_state:
            st.session_state['matchday_input'] = None
        if 'show_confirm' not in st.session_state:
            st.session_state['show_confirm'] = False
        if 'data_written' not in st.session_state:
            st.session_state['data_written'] = False

        if st.session_state['data_written']:
            st.success("Successfully wrote match result to database. Do you want to enter a new match result?")
            if st.button("Enter New Match Result"):
                reset_session_state()
                st.experimental_rerun()
        else:
            st.title("Racquet Records: Document your match results")
        
            st.write("Log your praiseworthy or pitiful match results here:")
            
            # Use selectbox for player names with an option to add a new player
            selected_player1 = st.selectbox("Player 1 Name", [''] + player_names + ['Add New Player'])
        
            if selected_player1 == 'Add New Player':
                new_player_name = st.text_input("Enter New Player Name")
                if new_player_name.strip() != '':
                    player_names.append(new_player_name.strip())
                    selected_player1 = new_player_name.strip()

            if selected_player1 != '':
                # Use number input for player 1 score with a default value of 0
                st.session_state['player1_name'] = selected_player1
                st.session_state['player1_score'] = st.number_input("Player 1 Score", min_value=0, value=st.session_state.get('player1_score', 0), step=1)
                
                if st.session_state['player1_score'] is not None:
                    # Use selectbox for player 2 name with an option to add a new player
                    selected_player2 = st.selectbox("Player 2 Name", [''] + player_names + ['Add New Player'])
        
                    if selected_player2 == 'Add New Player':
                        new_player_name = st.text_input("Enter New Player Name")
                        if new_player_name.strip() != '':
                            player_names.append(new_player_name.strip())
                            selected_player2 = new_player_name.strip()

                    if selected_player2 != '':
                        # Use number input for player 2 score with a default value of 0
                        st.session_state['player2_name'] = selected_player2
                        st.session_state['player2_score'] = st.number_input("Player 2 Score", min_value=0, value=st.session_state.get('player2_score', 0), step=1)
                        if st.session_state['player2_score'] is not None:
                            st.session_state['matchday_input'] = st.date_input("Matchday", st.session_state['matchday_input'] if st.session_state['matchday_input'] else None)
        
            if st.session_state['matchday_input'] and (st.session_state['player1_name'] or st.session_state['player2_name']):
                if st.button("Preview"):
                    st.subheader("Confirm the following match result:")
                    st.write(f"{st.session_state['player1_name']}: {st.session_state['player1_score']} - {st.session_state['player2_name']}: {st.session_state['player2_score']} on {st.session_state['matchday_input']}")
                    st.session_state['show_confirm'] = True
        
            if st.session_state['show_confirm']:
                if st.button("Confirm"):
                    spreadsheetId = st.secrets["public_gsheets_id"]
                    scope = ['https://www.googleapis.com/auth/spreadsheets']
                    google_creds_dict = dict(st.secrets["google_creds"])
                    # Create credentials from the dictionary
                    credentials = Credentials.from_service_account_info(google_creds_dict, scopes=scope)
                    client = gspread.authorize(credentials)
                    spreadsheet = client.open_by_key(spreadsheetId)
                    worksheet = spreadsheet.sheet1
                    match_result = f"{st.session_state['player1_name']} - {st.session_state['player2_name']} {st.session_state['player1_score']}:{st.session_state['player2_score']}"
                    game_date = int(st.session_state['matchday_input'].strftime('%Y%m%d'))
                    # Append the new game and its date to the worksheet
                    worksheet.append_row([match_result, game_date])
                    # Clear the inputs and flag data as written
                    st.session_state['data_written'] = True               
                    # Clear the inputs
                    st.session_state['player1_name'] = ''
                    st.session_state['player1_score'] = None
                    st.session_state['player2_name'] = ''
                    st.session_state['player2_score'] = None
                    st.session_state['matchday_input'] = None
                    st.session_state['show_confirm'] = False
        
                    st.experimental_rerun()  # This will rerun the script and update the UI with cleared inputs

    display_enter_match_results()
