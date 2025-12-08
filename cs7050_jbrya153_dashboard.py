import streamlit as st
import pandas as pd
import sqlite3
from io import StringIO
import numpy as np

# --- Configuration ---
DB_PATH = 'nba_archetype_data.db'

# Map internal column names to user-friendly display names for the profile box
PROFILE_MAPPING = {
    'Archetype_Label': 'Archetype Name',
    'Player_Count': 'Players Count',
    'PTS_Per_36': 'Avg. PTS / 36 Min',
    'REB_Per_36': 'Avg. REB / 36 Min',
    'AST_Per_36': 'Avg. AST / 36 Min',
    'Defensive_Score': 'Avg. DEF Score (Raw)',
    'MIN_Per_Game': 'Avg. MIN / Game',
    'HEIGHT': 'Avg. Height (in)',
    'WEIGHT': 'Avg. Weight (lbs)',
    'Shot_Pref_Three_Pointer': '3PT Preference (Volume)',
    'Shot_Pref_Layup_Dunk_Near_Rime': 'Layup/Dunk Preference',
    'Shot_Pref_Mid_Range_Jumper': 'Mid-Range Preference',
    'Shot_Percentage_Three_Pointer': '3PT Efficiency (%)',
    'Shot_Percentage_Layup_Dunk_Near_Rime': 'Layup/Dunk Efficiency (%)',
}
# Define the order of columns to be displayed in the cluster profile
PROFILE_COLUMNS_ORDER = [
    'Player_Count',
    'PTS_Per_36', 'AST_Per_36', 'REB_Per_36', 'Defensive_Score', 'MIN_Per_Game',
    'HEIGHT', 'WEIGHT',
    'Shot_Pref_Three_Pointer', 'Shot_Pref_Layup_Dunk_Near_Rime', 'Shot_Pref_Mid_Range_Jumper',
    'Shot_Percentage_Three_Pointer', 'Shot_Percentage_Layup_Dunk_Near_Rime',
]


# --- Data Loading ---
@st.cache_resource
def load_data():
    conn = sqlite3.connect(DB_PATH)
    # The structure of df_assignments must include all player columns.
    df_assignments = pd.read_sql('SELECT * FROM player_assignments', conn)
    # The structure of df_details must include all 14 cluster features.
    df_details = pd.read_sql('SELECT * FROM cluster_details', conn)
    df_details.set_index('index', inplace=True)
    conn.close()

    # --- NORMALIZATION (Global Max) ---
    max_def_score = df_assignments['Defensive_Score'].max()
    df_assignments['Defensive Score (0-100)'] = (
                                                        df_assignments['Defensive_Score'] / max_def_score
                                                ) * 100
    df_assignments['Defensive Score (0-100)'] = df_assignments['Defensive Score (0-100)'].round(1)

    return df_assignments, df_details


df_assignments, df_details = load_data()

st.set_page_config(layout="wide")
st.title("🏀 NBA Player Archetype Explorer")

# --- Sidebar Selection ---
all_archetypes = df_assignments['Archetype'].unique().tolist()

selected_archetypes = st.sidebar.multiselect(
    '1. Select Player Archetype(s):',
    options=all_archetypes,
    default=all_archetypes
)

player_search = st.sidebar.text_input(
    '2. Search Player Name (e.g., LeBron or Tatum):',
    ''
)

# --- Filtering ---
df_filtered_players = df_assignments[
    df_assignments['Archetype'].isin(selected_archetypes)
].sort_values(by='CORE_VALUE', ascending=False)

if player_search:
    df_filtered_players = df_filtered_players[
        df_filtered_players['DISPLAY_FIRST_LAST'].str.contains(player_search, case=False, na=False)
    ]

# Get details for the selected clusters
df_selected_details = df_details[
    df_details['Archetype_Label'].isin(selected_archetypes)
]

# --- Two-Column Layout Implementation ---

col1, col2 = st.columns([0.5, 0.5])

# -------------------------------------------------------------
# COLUMN 1: Cluster Details (Now displays all 14 features)
# -------------------------------------------------------------
with col1:
    st.header("Selected Profiles")
    st.markdown("---")

    if len(df_selected_details) > 0:
        st.subheader("Cluster Center Summaries")

        # Create a combined, formatted DataFrame for display
        df_profile_display = pd.DataFrame()

        for cluster_id in df_selected_details.index:
            row = df_selected_details.loc[cluster_id]

            # Select the required metrics in the desired order
            metrics = row[PROFILE_COLUMNS_ORDER].copy()

            # Formatting for display readability
            metrics_formatted = metrics.apply(lambda x: f"{x:.2f}" if isinstance(x, (float, np.floating)) else x)

            df_temp = pd.DataFrame({
                'Metric': [PROFILE_MAPPING[col] for col in PROFILE_COLUMNS_ORDER],
                row['Archetype_Label']: metrics_formatted.values
            })

            if df_profile_display.empty:
                df_profile_display = df_temp.set_index('Metric')
            else:
                df_profile_display = df_profile_display.join(df_temp.set_index('Metric'))

        # Display the profile details
        st.dataframe(df_profile_display, use_container_width=True)

    else:
        st.info("Select one or more archetypes in the sidebar.")

# -------------------------------------------------------------
# COLUMN 2: Filtered Player List (UNCHANGED)
# -------------------------------------------------------------
with col2:
    st.header("Assigned Players")
    st.subheader(f"Total Players Selected: {len(df_filtered_players)}")

    if len(df_filtered_players) == 0:
        st.warning("No players match the current filter and search criteria.")
    else:
        st.dataframe(
            df_filtered_players[[
                'DISPLAY_FIRST_LAST', 'CORE_VALUE', 'PTS_Per_36', 'AST_Per_36', 'REB_Per_36', 'Defensive Score (0-100)',
                'Archetype'
            ]],
            hide_index=True,
            use_container_width=True
        )