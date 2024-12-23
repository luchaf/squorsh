import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from pointless_utils import extract_data_from_games

# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)

# Read the data from the Google Sheet.
df_sheet = conn.read()

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
    # Process the data
    df_sheet["date"] = df_sheet["date"].astype(str)
    list_of_available_dates = list(set(df_sheet["date"].tolist()))
    st.dataframe(df_sheet)

    df_sheet = df_tmp.copy()

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
                    # Write the updated DataFrame back to the Google Sheet
                    conn.write(df)
                    st.success("Row inserted and data updated in Google Sheet.")
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
                    # Write the updated DataFrame back to the Google Sheet
                    conn.write(df)
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
        if st.button("Update"):
            try:
                df.at[selected_row, column_to_update] = new_value
                st.dataframe(df)
                # Write the updated DataFrame back to the Google Sheet
                conn.write(df)
                st.success("Value updated and data saved to Google Sheet.")
            except Exception as e:
                st.error(f"Error updating value: {str(e)}")

with online_form:
    player_names = ["Friedemann", "Lucas", "Peter", "Simon", "Tobias"]

    def reset_session_state():
        """Helper function to reset session state."""
        for key in ['player1_name', 'player1_score', 'player2_name', 'player2_score', 'matchday_input', 'show_confirm', 'data_written']:
            st.session_state[key] = None

    def display_enter_match_results():
        if 'data_written' not in st.session_state:
            st.session_state['data_written'] = False

        if st.session_state['data_written']:
            st.success("Successfully wrote match result to database. Do you want to enter a new match result?")
            if st.button("Enter New Match Result"):
                reset_session_state()
                st.experimental_rerun()
        else:
            st.title("Racquet Records: Document your match results")

            player1_name = st.selectbox("Player 1 Name", [''] + player_names)
            player1_score = st.number_input("Player 1 Score", min_value=0, step=1)
            player2_name = st.selectbox("Player 2 Name", [''] + player_names)
            player2_score = st.number_input("Player 2 Score", min_value=0, step=1)
            matchday_input = st.date_input("Matchday", date.today())

            if st.button("Submit"):
                new_data = {
                    "Player1": player1_name,
                    "Score1": player1_score,
                    "Player2": player2_name,
                    "Score2": player2_score,
                    "date": matchday_input.strftime('%Y-%m-%d')
                }
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                # Write the new data to the Google Sheet
                conn.write(df)
                st.success("Match result saved!")
                st.session_state['data_written'] = True

    display_enter_match_results()
