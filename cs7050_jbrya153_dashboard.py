import streamlit as st
import pandas as pd
import sqlite3

# --- Configuration ---
DB_PATH = 'nba_archetype_data.db' 

# --- Data Loading (Simplified to only fetch player data) ---
@st.cache_resource
def load_data():
    conn = sqlite3.connect(DB_PATH)
    # Load all player assignments (which contains all calculated metrics)
    df_assignments = pd.read_sql('SELECT * FROM player_assignments', conn)
    conn.close()

    # --- NORMALIZATION (Global Max) ---
    # Ensure defensive score is normalized, even without the cluster context
    max_def_score = df_assignments['Defensive_Score'].max()
    df_assignments['Defensive Score (0-100)'] = (
        df_assignments['Defensive_Score'] / max_def_score
    ) * 100
    df_assignments['Defensive Score (0-100)'] = df_assignments['Defensive Score (0-100)'].round(1)
    
    return df_assignments

df_assignments = load_data()


st.set_page_config(layout="wide")
st.title("🏀 NBA Player List Explorer")

# --- Sidebar Selection (Only Player Search remains) ---

player_search = st.sidebar.text_input(
    'Search Player Name (e.g., LeBron or Tatum):',
    '' 
)

# --- Filtering ---
# Start with all players, sorted by the core ranking value
df_filtered_players = df_assignments.sort_values(by='CORE_VALUE', ascending=False)

# Filter by Search Text (case-insensitive and only applies if text is entered)
if player_search:
    df_filtered_players = df_filtered_players[
        df_filtered_players['DISPLAY_FIRST_LAST'].str.contains(player_search, case=False, na=False)
    ]


# --- Main Display (Single Column) ---
st.header("Filtered Player List")
st.subheader(f"Total Players Displayed: {len(df_filtered_players)}")

if len(df_filtered_players) == 0:
    st.warning("No players match the current search criteria.")
else:
    # Display the essential player data columns
    st.dataframe(
        df_filtered_players[[
            'DISPLAY_FIRST_LAST', 
            'CORE_VALUE', 
            'PTS_Per_36', 
            'AST_Per_36', 
            'REB_Per_36', 
            'Defensive Score (0-100)',
            'POSITION', # Keeping physical/positional info
            'MIN_Per_Game' 
        ]],
        hide_index=True,
        use_container_width=True
    )