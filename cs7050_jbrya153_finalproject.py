import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import commonplayerinfo, leaguedashplayerstats, shotchartdetail
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import sys
import sqlite3

# Suppress FutureWarning from scikit-learn/KMeans
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
# ADJUSTABLE VARIABLE: Number of players to analyze (set to 20 per your test)
TOP_X_PLAYERS = 25

# ADJUSTABLE VARIABLE: Number of clusters (set to 4 per your test)
N_CLUSTERS = 10

# Set the desired range of years for the project (1990-91 to 2024-25)
#START_YEAR_PROJECT = 1990
#END_YEAR_PROJECT = 2024

# For efficient testing/runtime, set the run range here:
START_YEAR_RUN = 2024
END_YEAR_RUN = 2024

# API Call Management
SLEEP_TIME = 0.2
MAX_WORKERS = 8
DB_PATH = 'nba_archetype_data.db'  # Database file path


# --- Helper Functions ---

def get_season_list(start_year, end_year):
    """Generates a list of season strings (e.g., '2023-24')."""
    seasons = []
    for year in range(start_year, end_year + 1):
        season_str = f"{year}-{str(year + 1)[2:]}"
        seasons.append(season_str)
    return seasons


def convert_height_to_inches(height_str):
    """Converts height string ('X-Y') to total inches."""
    if pd.isna(height_str) or height_str is None:
        return np.nan
    if isinstance(height_str, (int, float)):
        return height_str
    try:
        feet, inches = map(int, height_str.split('-'))
        return (feet * 12) + inches
    except:
        return np.nan


# --- 1. Fast Aggregated Seasonal Stats Retrieval (OPTIMIZED & CONCURRENT READY) ---
def fetch_all_seasonal_stats_fast(season):
    """
    Fetches all player seasonal stats, calculates the Core Value,
    and returns only the top X players.
    """
    try:
        player_stats_dashboard = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star='Regular Season',
        )
        time.sleep(SLEEP_TIME)

        df_seasonal_stats = player_stats_dashboard.get_data_frames()[0]

        # 1. Filter for players with minimum 10 games played
        df_seasonal_stats = df_seasonal_stats[df_seasonal_stats['GP'] > 10].copy()

        # 2. Calculate the core ranking value (PTS + AST + REB + STL + BLK)
        df_seasonal_stats.loc[:, 'CORE_VALUE'] = (
                df_seasonal_stats['PTS'] +
                df_seasonal_stats['AST'] +
                df_seasonal_stats['REB'] +
                df_seasonal_stats['STL'] +
                df_seasonal_stats['BLK']
        )

        # 3. Sort and filter down to the TOP_X_PLAYERS
        df_seasonal_stats = df_seasonal_stats.sort_values(
            by='CORE_VALUE', ascending=False
        ).head(TOP_X_PLAYERS).copy()

        df_seasonal_stats['SEASON_ID'] = season

        return df_seasonal_stats

    except Exception:
        return pd.DataFrame()


# --- 2. Fetch General Player Details (Static Info - CONCURRENT) ---
def fetch_player_general_details(player_ids):
    """Fetches non-statistical details for the filtered target players using concurrency."""
    all_details = []

    def fetch_single_player_details(player_id):
        try:
            info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            time.sleep(SLEEP_TIME / MAX_WORKERS)
            info_df = info.get_data_frames()[0]

            relevant_columns = [
                'PERSON_ID', 'DISPLAY_FIRST_LAST', 'HEIGHT', 'WEIGHT',
                'POSITION', 'DRAFT_ROUND', 'DRAFT_NUMBER'
            ]
            return info_df[relevant_columns]
        except Exception:
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_player = {
            executor.submit(fetch_single_player_details, player_id): player_id
            for player_id in player_ids
        }

        for future in tqdm(as_completed(future_to_player), total=len(player_ids),
                           desc="Fetching Player Details (ID, Pos, Height, Weight)"):
            details_df = future.result()
            if not details_df.empty:
                all_details.append(details_df)

    if all_details:
        df_player_details = pd.concat(all_details, ignore_index=True)
        # Rename PERSON_ID to PLAYER_ID for merging
        df_player_details = df_player_details.drop_duplicates(subset=['PERSON_ID']).rename(
            columns={'PERSON_ID': 'PLAYER_ID'}
        )
    else:
        df_player_details = pd.DataFrame()

    return df_player_details


# --- 3. CONCURRENT Fetch and Process Shot Chart Data ---

def classify_shot_type(row):
    """Helper function to classify shot types consistently."""
    if row['SHOT_ZONE_RANGE'] == '24+ ft.': return 'Three_Pointer'
    if 'Dunk' in row['ACTION_TYPE'] or 'Layup' in row['ACTION_TYPE']: return 'Layup_Dunk_Near_Rime'
    if row['SHOT_ZONE_RANGE'] in ['8-16 ft.', '16-24 ft.']:
        if 'Jump Shot' in row['ACTION_TYPE']: return 'Mid_Range_Jumper'
    if row['SHOT_ZONE_RANGE'] == 'Less Than 8 ft.': return 'Near_Rim_Other'
    return 'Other'


def fetch_single_player_shot_chart(player_id, season):
    """Worker function to fetch shot chart for a single player."""
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            context_measure_simple='FGA',
            season_nullable=season,
            season_type_all_star='Regular Season'
        )
        time.sleep(SLEEP_TIME / MAX_WORKERS)
        shots_df = shot_data.get_data_frames()[0]

        if not shots_df.empty:
            shots_df['SEASON_ID'] = season
            shots_df['PLAYER_ID'] = player_id
            shots_df['SHOT_CATEGORY'] = shots_df.apply(classify_shot_type, axis=1)
            return shots_df

    except Exception:
        return pd.DataFrame()


def fetch_and_process_shot_chart_data_concurrent(player_ids, season):
    """
    Fetches shot chart data for the single season CONCURRENTLY and calculates
    seasonal shot preferences and percentages.
    """
    all_shots = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_player = {
            executor.submit(fetch_single_player_shot_chart, player_id, season): player_id
            for player_id in player_ids
        }

        for future in tqdm(as_completed(future_to_player), total=len(player_ids),
                           desc=f"Fetching {season} Shot Chart Data (Concurrent)"):
            result_df = future.result()
            if not result_df.empty:
                all_shots.append(result_df)

    if not all_shots:
        return pd.DataFrame()

    df_all_shots = pd.concat(all_shots, ignore_index=True)

    # Aggregate shot data by player and season
    df_shot_agg = df_all_shots.groupby(['PLAYER_ID', 'SEASON_ID', 'SHOT_CATEGORY']).agg(
        Total_Attempts=('SHOT_ATTEMPTED_FLAG', 'sum'),
        Total_Makes=('SHOT_MADE_FLAG', 'sum')
    ).reset_index()

    # Calculate percentages
    df_shot_agg['Shot_Percentage'] = np.where(
        df_shot_agg['Total_Attempts'] > 0,
        df_shot_agg['Total_Makes'] / df_shot_agg['Total_Attempts'],
        0
    )

    # Pivot the data to wide format
    df_shot_pivot = df_shot_agg.pivot_table(
        index=['PLAYER_ID', 'SEASON_ID'],
        columns='SHOT_CATEGORY',
        values=['Total_Attempts', 'Shot_Percentage']
    ).fillna(0)

    # Flatten the multi-index columns
    df_shot_pivot.columns = [f'{col[0]}_{col[1]}' for col in df_shot_pivot.columns.values]
    df_shot_pivot = df_shot_pivot.reset_index()

    return df_shot_pivot


# --- 4. Main Data Collection and Merging Loop ---

SEASONS_TO_FETCH = get_season_list(START_YEAR_RUN, END_YEAR_RUN)


def process_single_season(season):
    """Fetches, processes, and merges data for a single season."""

    # 1. Fetch Seasonal Stats (AND FILTER TO TOP X)
    df_season_stats = fetch_all_seasonal_stats_fast(season)

    if df_season_stats.empty:
        return None, set()

    TARGET_PLAYER_IDS = set(df_season_stats['PLAYER_ID'].unique())

    # 3. Fetch and Process Shot Chart Data (CONCURRENTLY on only the TOP X)
    df_shot_chart_seasonal = fetch_and_process_shot_chart_data_concurrent(TARGET_PLAYER_IDS, season)

    # Merge Seasonal Stats and Shot Chart Data
    df_data = pd.merge(
        df_season_stats,
        df_shot_chart_seasonal,
        on=['PLAYER_ID', 'SEASON_ID'],
        how='left'
    ).fillna(0).copy()

    return df_data, TARGET_PLAYER_IDS


def fetch_all_data_concurrent(seasons):
    """Runs data fetching for multiple seasons concurrently."""
    all_player_stats_seasons = []
    all_player_details_ids = set()

    if len(seasons) > 1:
        print(f"Starting CONCURRENT processing for {len(seasons)} seasons...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_season = {
                executor.submit(process_single_season, season): season
                for season in seasons
            }

            for future in as_completed(future_to_season):
                try:
                    df_data, player_ids = future.result()
                    if df_data is not None:
                        all_player_stats_seasons.append(df_data)
                        all_player_details_ids.update(player_ids)
                except Exception:
                    pass
    else:
        # If only one season, run sequentially
        df_data, player_ids = process_single_season(seasons[0])
        if df_data is not None:
            all_player_stats_seasons.append(df_data)
            all_player_details_ids.update(player_ids)

    return all_player_stats_seasons, all_player_details_ids


# EXECUTE CONCURRENT FETCH
all_player_stats_seasons, all_player_details_ids = fetch_all_data_concurrent(SEASONS_TO_FETCH)

if not all_player_stats_seasons:
    sys.exit()

df_raw_seasonal_data = pd.concat(all_player_stats_seasons, ignore_index=True)

# 2. Fetch General Player Details (STATIC INFO - on all unique players)
df_player_details = fetch_player_general_details(all_player_details_ids)

# --- 5. Final Data Cleaning and Feature Engineering ---

df_data_for_clustering = df_raw_seasonal_data.copy()

# ... (Previous filtering and renaming logic is here) ...

# Ensure PLAYER_ID is a regular column in the seasonal stats DataFrame
if 'PLAYER_ID' not in df_data_for_clustering.columns:
    df_data_for_clustering = df_data_for_clustering.reset_index(names=['PLAYER_ID'])

# Ensure PLAYER_ID is a regular column in the player details DataFrame
if 'PLAYER_ID' not in df_player_details.columns:
    df_player_details = df_player_details.reset_index(names=['PLAYER_ID'])

# **ERROR LINE FIX (Line 314 is approximately here):**
df_final_data = pd.merge(
    df_data_for_clustering,
    df_player_details,
    on='PLAYER_ID', # This column is now guaranteed to be present in both
    how='left'
).drop_duplicates(subset=['PLAYER_ID', 'SEASON_ID'])

# 1. FIX: RESOLVE MISSING NAMES (NaNs)
# Use the robust dashboard name to fill any missing names from the concurrent API call failure.
df_final_data.loc[:, 'DISPLAY_FIRST_LAST'] = df_final_data['DISPLAY_FIRST_LAST'].fillna(
    df_final_data['PLAYER_NAME']
)

# 2. FIX: CONVERT HEIGHT/WEIGHT TO NUMERIC BEFORE IMPUTATION
df_final_data.loc[:, 'HEIGHT'] = df_final_data['HEIGHT'].apply(convert_height_to_inches)
df_final_data.loc[:, 'WEIGHT'] = pd.to_numeric(df_final_data['WEIGHT'], errors='coerce')

# 3. IMPUTATION: Fill NaNs for clustering features (including HEIGHT/WEIGHT)
df_final_data[['HEIGHT', 'WEIGHT']] = df_final_data[['HEIGHT', 'WEIGHT']].fillna(
    df_final_data[['HEIGHT', 'WEIGHT']].mean()
)

# 4. FEATURE ENGINEERING (creating per-36 min stats)
for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
    df_final_data.loc[:, f'{col}_Per_36'] = (df_final_data[col] / df_final_data['MIN']) * 36

df_final_data.loc[:, 'MIN_Per_Game'] = df_final_data['MIN'] / df_final_data['GP']
df_final_data.loc[:, 'Defensive_Score'] = df_final_data['STL_Per_36'] + df_final_data[
    'BLK_Per_36']

# Total Shot Attempts for Shot Preference calculation
total_shots_col = [col for col in df_final_data.columns if 'Total_Attempts_' in col]
df_final_data.loc[:, 'Total_FGA_ShotChart'] = df_final_data[total_shots_col].sum(axis=1)

# Shot Preference Ratios
for shot_type in ['Three_Pointer', 'Layup_Dunk_Near_Rime', 'Mid_Range_Jumper']:
    attempt_col = f'Total_Attempts_{shot_type}'
    df_final_data.loc[:, f'Shot_Pref_{shot_type}'] = np.where(
        df_final_data['Total_FGA_ShotChart'] > 0,
        df_final_data[attempt_col] / df_final_data['Total_FGA_ShotChart'],
        0
    )


# --- 6. K-Means Clustering Implementation and Labeling ---

def generate_archetype_label(row, cluster_centers):
    """
    Generates a descriptive archetype label for a cluster based on feature prominence.
    This function will be applied to the cluster centers.
    """
    cluster_id = row.name
    features = cluster_centers.loc[cluster_id]

    # 1. Base Role (Position-Based Scoring/Playmaking)
    base_role = ""
    if features['PTS_Per_36'] > cluster_centers['PTS_Per_36'].mean() * 1.15:
        base_role = "Offensive Star"
    elif features['AST_Per_36'] > cluster_centers['AST_Per_36'].mean() * 1.15 and features['REB_Per_36'] > cluster_centers['REB_Per_36'].mean() * 1.15:
        base_role = "Playmaking Rebounder"
    elif features['AST_Per_36'] > cluster_centers['AST_Per_36'].mean() * 1.15:
        base_role = "Playmaker"
    elif features['REB_Per_36'] > cluster_centers['REB_Per_36'].mean() * 1.15:
        base_role = "Rebounder"
    else:
        base_role = "Balanced Star"

    # 2. Defensive Tag (Requested Criteria)
    defensive_tag = ""
    avg_def_score = cluster_centers['Defensive_Score'].mean()
    if features['Defensive_Score'] > avg_def_score * 1.15:
        defensive_tag = "2-Way"

    # 3. Scoring Style Tag (Requested Criteria)
    style_tag = ""

    # Check for High 3-Point Percentage AND High Preference
    if ((features['Shot_Percentage_Three_Pointer'] > cluster_centers['Shot_Percentage_Three_Pointer'].mean()) * 0.8
            and (
            features['Shot_Pref_Three_Pointer'] > cluster_centers['Shot_Pref_Three_Pointer'].mean() * 1.2
    )):
        style_tag = "3-Point Shooting"

    # Check for High Volume of Layups/Dunks (Slashing/Interior Scoring)
    elif (features['Shot_Pref_Layup_Dunk_Near_Rime'] > cluster_centers['Shot_Pref_Layup_Dunk_Near_Rime'].mean() * 1.25):
        # Infer Interior Scoring for taller players (C, F/C heavy clusters)
        if features['HEIGHT'] > cluster_centers['HEIGHT'].mean():
            style_tag = "Interior Scoring"
        else:
            style_tag = "Slashing"

    # Check for High Mid-Range Preference
    elif (features['Shot_Pref_Mid_Range_Jumper'] > cluster_centers['Shot_Pref_Mid_Range_Jumper'].mean() * 1.25):
        style_tag = "Mid-Range"

    # Check for Rebounding Tag (Added for better description)
    elif (features['REB_Per_36'] > cluster_centers['REB_Per_36'].mean() * 1.3):
        style_tag = "Glass Cleaning"

    # Combine and Clean up the Label
    parts = [defensive_tag, style_tag, base_role]
    # Filter out empty strings and duplicate tags
    label_parts = [p for p in parts if p]

    # Simple formatting: e.g., '2-Way Primary Scorer (3-Point Shooting)'
    if len(label_parts) > 2:
        final_label = f"{label_parts[0]} {label_parts[1]} {label_parts[2]}"
    elif len(label_parts) == 2:
        final_label = f"{label_parts[0]} {label_parts[1]}"
    else:
        final_label = label_parts[0] if label_parts else "All-Around Role Player"

    return final_label


def run_kmeans_clustering(df, n_clusters):
    """
    Performs K-Means clustering, assigns archetypes based on cluster centers,
    and returns the clustered DataFrame and the cluster centers details.
    """

    # Final feature set for clustering: rate stats, efficiency, shot profile, and physicals
    features_to_cluster = [
        'PTS_Per_36', 'REB_Per_36', 'AST_Per_36', 'Defensive_Score',
        'Shot_Pref_Three_Pointer', 'Shot_Pref_Layup_Dunk_Near_Rime', 'Shot_Pref_Mid_Range_Jumper',
        'Shot_Percentage_Three_Pointer', 'Shot_Percentage_Layup_Dunk_Near_Rime',
        'HEIGHT', 'WEIGHT', 'MIN_Per_Game'
    ]

    # Drop any rows with NaN values in the features (should be handled by fillna earlier, but safety check)
    df_clust = df.dropna(subset=features_to_cluster).copy()

    # If the number of players is less than n_clusters, reduce n_clusters
    if len(df_clust) < n_clusters:
        n_clusters = len(df_clust) // 2
        print(f"Warning: Player count ({len(df_clust)}) is too low. Reducing k to {n_clusters}.")
        if n_clusters < 2:
            print("Fatal Error: Insufficient player data for clustering. Exiting.")
            sys.exit()

    df_features = df_clust[features_to_cluster].copy()

    # Standardization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    # Running K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clust.loc[:, 'Cluster'] = kmeans.fit_predict(scaled_features)

    # Analyze cluster centers (in original scale) for labeling
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features_to_cluster
    )

    # Add player counts to the center analysis
    cluster_counts = df_clust.groupby('Cluster')['PLAYER_ID'].count().rename('Player_Count')
    cluster_centers = cluster_centers.join(cluster_counts)

    # Generate and assign archetype labels
    cluster_centers['Archetype_Label'] = cluster_centers.apply(
        lambda row: generate_archetype_label(row, cluster_centers), axis=1
    )

    # Map the labels back to the player data
    label_map = cluster_centers['Archetype_Label'].to_dict()
    df_clust.loc[:, 'Archetype'] = df_clust['Cluster'].map(label_map)

    return df_clust, cluster_centers


# --- Main Execution ---

# EXECUTE CONCURRENT FETCH
all_player_stats_seasons, all_player_details_ids = fetch_all_data_concurrent(SEASONS_TO_FETCH)

if not all_player_stats_seasons:
    sys.exit()

df_raw_seasonal_data = pd.concat(all_player_stats_seasons, ignore_index=True)

# 2. Fetch General Player Details (STATIC INFO - on all unique players)
df_player_details = fetch_player_general_details(all_player_details_ids)

# --- 5. Final Data Cleaning and Feature Engineering (INCLUDES ERROR FIX) ---

df_data_for_clustering = df_raw_seasonal_data.copy()

# Filter out players who had 0 minutes played in the dashboard
df_data_for_clustering = df_data_for_clustering[df_data_for_clustering['MIN'] > 0].copy()

# Rename the reliable player name from the dashboard for backup
df_data_for_clustering = df_data_for_clustering.rename(columns={'PLAYER_NAME': 'NAME_DASHBOARD_BACKUP'})

# Merge with General Details (Static Info)
df_final_data = pd.merge(
    df_data_for_clustering,
    df_player_details,
    on='PLAYER_ID',
    how='left'
).drop_duplicates(subset=['PLAYER_ID', 'SEASON_ID'])

# 1. FIX: RESOLVE MISSING NAMES (NaNs)
df_final_data.loc[:, 'DISPLAY_FIRST_LAST'] = df_final_data['DISPLAY_FIRST_LAST'].fillna(
    df_final_data['NAME_DASHBOARD_BACKUP']
)

# 2. FIX: CONVERT HEIGHT/WEIGHT TO NUMERIC BEFORE IMPUTATION
df_final_data.loc[:, 'HEIGHT'] = df_final_data['HEIGHT'].apply(convert_height_to_inches)
df_final_data.loc[:, 'WEIGHT'] = pd.to_numeric(df_final_data['WEIGHT'], errors='coerce')

# 3. IMPUTATION: Fill NaNs for clustering features (including HEIGHT/WEIGHT)
df_final_data[['HEIGHT', 'WEIGHT']] = df_final_data[['HEIGHT', 'WEIGHT']].fillna(
    df_final_data[['HEIGHT', 'WEIGHT']].mean()
)

# 4. FEATURE ENGINEERING (creating per-36 min stats)
for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
    df_final_data.loc[:, f'{col}_Per_36'] = (df_final_data[col] / df_final_data['MIN']) * 36

df_final_data.loc[:, 'MIN_Per_Game'] = df_final_data['MIN'] / df_final_data['GP']
df_final_data.loc[:, 'Defensive_Score'] = df_final_data['STL_Per_36'] + df_final_data[
    'BLK_Per_36']

# Total Shot Attempts for Shot Preference calculation
total_shots_col = [col for col in df_final_data.columns if 'Total_Attempts_' in col]
df_final_data.loc[:, 'Total_FGA_ShotChart'] = df_final_data[total_shots_col].sum(axis=1)

# Shot Preference Ratios
for shot_type in ['Three_Pointer', 'Layup_Dunk_Near_Rime', 'Mid_Range_Jumper']:
    attempt_col = f'Total_Attempts_{shot_type}'
    df_final_data.loc[:, f'Shot_Pref_{shot_type}'] = np.where(
        df_final_data['Total_FGA_ShotChart'] > 0,
        df_final_data[attempt_col] / df_final_data['Total_FGA_ShotChart'],
        0
    )

# --- 7. K-Means Clustering and Data Saving ---

df_clustered_seasonal_data, df_cluster_details = run_kmeans_clustering(df_final_data.copy(), n_clusters=N_CLUSTERS)

# Define the full set of columns to save for the interactive dashboard
PLAYER_COLUMNS_TO_SAVE = [
    'PLAYER_ID', 'DISPLAY_FIRST_LAST', 'SEASON_ID', 'POSITION',
    'HEIGHT', 'WEIGHT', 'GP', 'MIN',

    # Core Total Stats & Ranking Metric
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'CORE_VALUE',

    # Per-36 Rate Stats
    'PTS_Per_36', 'REB_Per_36', 'AST_Per_36', 'Defensive_Score',

    # Shot Profile Metrics
    'Shot_Pref_Three_Pointer', 'Shot_Pref_Layup_Dunk_Near_Rime', 'Shot_Pref_Mid_Range_Jumper',
    'Shot_Percentage_Three_Pointer', 'Shot_Percentage_Layup_Dunk_Near_Rime', 'Shot_Percentage_Mid_Range_Jumper',

    # Clustering Results
    'Archetype', 'Cluster'
]

CLUSTER_FEATURES = [
    'Archetype_Label', 'Player_Count', 'PTS_Per_36', 'REB_Per_36', 'AST_Per_36', 'Defensive_Score',
    'Shot_Pref_Three_Pointer', 'Shot_Pref_Layup_Dunk_Near_Rime', 'Shot_Pref_Mid_Range_Jumper',
    'Shot_Percentage_Three_Pointer', 'Shot_Percentage_Layup_Dunk_Near_Rime',
    'HEIGHT', 'WEIGHT', 'MIN_Per_Game'
]

# --- Data Persistence ---
conn = sqlite3.connect(DB_PATH)

# Save 1: Player Assignments (All details required for dashboard filtering)
df_clustered_seasonal_data[PLAYER_COLUMNS_TO_SAVE].to_sql(
    'player_assignments', conn, if_exists='replace', index=False
)

# Save 2: Cluster Definitions (Details of the k=4 archetypes)
#df_cluster_details[['Archetype_Label', 'Player_Count',
#                    'PTS_Per_36', 'REB_Per_36', 'Defensive_Score',
#                    'Shot_Pref_Three_Pointer']].to_sql(
#   'cluster_details', conn, if_exists='replace', index=True
#)

df_cluster_details[CLUSTER_FEATURES].to_sql(
    'cluster_details', conn, if_exists='replace', index=True
)

conn.close()
print(f"\nResults successfully saved to {DB_PATH}")

print("\n--- Final Cluster Details ---")
print(df_cluster_details[['Archetype_Label', 'Player_Count', 'PTS_Per_36', 'Defensive_Score']].to_string())