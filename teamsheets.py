from collections import Counter
import pandas as pd
import streamlit as st
import warnings
import logging

warnings.filterwarnings("ignore")

# ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_team_profile(team_name, dataframe):
    """
    Get the team profile for the selected team.

    Parameters:
    - team_name: The name of the team.
    - dataframe: The filtered DataFrame containing game lineup information.

    Returns:
    - A DataFrame with the team profile information.
    """
    import scipy.stats as stats
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules

    # Filter for the selected team
    team_data = dataframe[dataframe["team"] == team_name]

    # Get the most common positions associations
    positions_data = team_data.groupby("game_id")["most_common_position"].agg(list)
    positions_data = positions_data.reset_index()

    # Filter the games where the number of positions is exactly 10
    positions_data = positions_data[
        positions_data["most_common_position"].apply(len) == 10
    ]

    positions_data["most_common_position"] = positions_data[
        "most_common_position"
    ].apply(lambda x: sorted(x))

    min_support = 0.1

    # Convert the list of positions into a DataFrame of boolean values
    positions_data_bool = (
        positions_data["most_common_position"].str.join("|").str.get_dummies()
    )

    # Run the Apriori algorithm
    while min_support > 0:
        frequent_itemsets = apriori(
            positions_data_bool, min_support=min_support, use_colnames=True
        )
        if frequent_itemsets.empty:
            min_support -= 0.01
        else:
            break

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    rules = rules.sort_values(by="confidence", ascending=False)

    return rules


def get_positions_of_each_game(fbref_lineups, team_name):
    # get teams from the specified team that is_starter is true
    order = ["LB", "RB", "WB", "CB", "LM", "RM", "DM", "CM", "MF", "AM", "FW"]

    team_starters = fbref_lineups[
        (fbref_lineups["team"] == team_name) & (fbref_lineups["is_starter"] == True)
    ]

    # rename to most_common_position to starters_positions
    team_starters = team_starters.rename(
        columns={"most_common_position": "starters_positions"}
    )

    # group the players so we get position as a list, and sum the is_oop column
    positions_data = team_starters.groupby("game_id").agg(
        {"starters_positions": list, "is_oop": "sum"}
    )

    # reset index
    positions_data = positions_data.reset_index()

    # sort starters_positions according to the order list
    positions_data["starters_positions"] = positions_data["starters_positions"].apply(
        lambda x: sorted(
            x,
            key=lambda position: (
                order.index(position) if position in order else len(order)
            ),
        )
    )

    # convert list to tuple
    positions_data["starters_positions"] = positions_data["starters_positions"].apply(
        tuple
    )

    # group by starters_positions and aggregate is_oop and count
    positions_data = (
        positions_data.groupby("starters_positions")
        .agg({"is_oop": "mean", "game_id": "count"})
        .reset_index()
    )

    # rename game_id to count
    positions_data = positions_data.rename(columns={"game_id": "count"})

    # make sure count is greater than 1
    positions_data = positions_data[positions_data["count"] > 1]

    # sort by count in descending order
    positions_data = positions_data.sort_values(
        by="count", ascending=False
    ).reset_index(drop=True)

    return positions_data



# create a function to look for the positions a player passed has played and count
def get_player_positions(fbref_lineups, player_name, team_name):
    # get teams from the specified team that is_starter is true
    team_starters = fbref_lineups[
        (fbref_lineups["team"] == team_name) & (fbref_lineups["is_starter"] == True)
    ]

    # filter player names that contain the partial player name
    filtered_players = team_starters[
        team_starters["player"].str.contains(player_name, case=False)
    ]

    player_match = filtered_players["player"].unique().astype(str)

    print(f"Player match: {player_match}\n\n")

    # get the unique positions
    positions = filtered_players["position"].unique().tolist()

    # get the number of games played
    num_games = filtered_players.shape[0]

    # Initialize an empty DataFrame for position counts
    position_counts = pd.DataFrame(
        columns=[
            "Position",
            "Count",
            "Most Recent Date",
            "Other Players",
            "Home Games",
            "Away Games",
        ]
    )

    # Iterate over each position and count the occurrences
    for position in positions:
        position_data = filtered_players[filtered_players["position"] == position]
        count = position_data.shape[0]
        most_recent_date = position_data["date"].max()

        other_players = (
            team_starters[team_starters["game"].isin(position_data["game"])]["player"]
            .unique()
            .tolist()
        )
        if player_name in other_players:
            other_players.remove(player_name)

        home_games = position_data[position_data["home_team"] == team_name].shape[0]
        away_games = position_data[position_data["away_team"] == team_name].shape[0]

        # Using pandas.concat instead of append
        new_row = pd.DataFrame(
            [
                {
                    "Position": position,
                    "Count": count,
                    "Most Recent Date": most_recent_date,
                    "Other Players": other_players,
                    "Home Games": home_games,
                    "Away Games": away_games,
                }
            ]
        )
        position_counts = pd.concat([position_counts, new_row], ignore_index=True)

    # sort the position counts by count in descending order and reset index
    position_counts = position_counts.sort_values(
        by="Count", ascending=False
    ).reset_index(drop=True)

    # sort the position counts by count in descending order
    position_counts = position_counts.sort_values(
        by="Count", ascending=False
    ).reset_index(drop=True)

    # get the opponents for each game
    opponents = (
        filtered_players[filtered_players["game"].isin(filtered_players["game"])]
        .groupby("opponent")
        .size()
        .reset_index(name="Count")
        .sort_values(by="Count", ascending=False)
        .reset_index(drop=True)
    )

    return position_counts, opponents


def get_player_positions_v2(fbref_lineups, player_name, team_name):
    logging.info(f"Getting player positions for {player_name} from {team_name}")

    # Filter for the specific team and players that contain the player_name
    team_data = fbref_lineups[
        (fbref_lineups["team"] == team_name)
        & (fbref_lineups["player"].str.contains(player_name, case=False))
        & (fbref_lineups["is_starter"] == True)
    ]
    logging.info(f"Filtered team data: {team_data.head()}")

    # Initialize an empty dictionary to hold position counts
    position_counts_dict = {}

    # Position columns to consider
    position_columns = ["position_1", "position_2", "position_3", "position_4"]

    # Iterate through each row and each position column to count positions
    for _, row in team_data.iterrows():
        for col in position_columns:
            pos = row[col]
            if pd.notnull(pos):  # Check if the position value is not NaN
                if pos in position_counts_dict:
                    position_counts_dict[pos] += 1
                else:
                    position_counts_dict[pos] = 1

    logging.info(f"Position counts: {position_counts_dict}")

    # Convert the dictionary to a DataFrame
    position_counts_df = pd.DataFrame(
        list(position_counts_dict.items()), columns=["Position", "Count"]
    )

    # Calculate total count of positions
    total_count = position_counts_df["Count"].sum()

    # Calculate percentage for each position
    position_counts_df["Percentage"] = (position_counts_df["Count"] / total_count) * 100

    # Sort the DataFrame by count in descending order
    position_counts_df = position_counts_df.sort_values(
        by="Count", ascending=False
    ).reset_index(drop=True)

    logging.info(f"Position counts DataFrame: \n{position_counts_df}")

    # Opponents faced analysis
    opponents = (
        team_data.groupby("opponent")
        .size()
        .reset_index(name="Count")
        .sort_values(by="Count", ascending=False)
        .reset_index(drop=True)
    )

    # Calculate total count of opponents
    total_opponents = opponents["Count"].sum()

    # Calculate percentage for each opponent
    opponents["Percentage"] = (opponents["Count"] / total_opponents) * 100

    # log unique opponents values
    logging.info(f"Unique opponents: {opponents['opponent'].unique()}")

    logging.info(f"Opponents DataFrame: \n{opponents}")

    # Get opponents when player is not a starter
    non_starter_data = fbref_lineups[
        (fbref_lineups["team"] == team_name)
        & (fbref_lineups["player"].str.contains(player_name, case=False))
        & (fbref_lineups["is_starter"] == False)
    ]
    non_starter_opponents = non_starter_data["opponent"].unique()
    logging.info(f"Opponents when {player_name} is not a starter: {non_starter_opponents}")

    return position_counts_df, opponents, non_starter_opponents

def get_most_common_players(
    team_name, selected_players, excluded_players, dataframe, set_piece_takers=False
):
    # Ensure selected_players and excluded_players are lists
    if not isinstance(selected_players, list):
        selected_players = [selected_players]
    if not isinstance(excluded_players, list):
        excluded_players = [excluded_players]

    logging.info(f"Filtering for {team_name}. Including: {selected_players}. Excluding: {excluded_players}\n\n")

    # Drop duplicates based on 'game_id' and 'player' columns
    dataframe = dataframe.drop_duplicates(subset=['game_id', 'player'])

    # Filter for is_starter == True and for the selected team
    dataframe = dataframe[dataframe["is_starter"] == True]
    team_data = dataframe[dataframe["team"] == team_name]

    # Function to filter games based on selected and excluded players
    def game_filter(players):
        return set(selected_players).issubset(set(players)) and set(
            excluded_players
        ).isdisjoint(set(players))

    # Apply the game filter
    games_with_selected_players = (
        team_data.groupby("game_id")["player"].apply(list).apply(game_filter)
    )
    logging.info(f"Games with selected players: {games_with_selected_players.head()}")

    valid_games = games_with_selected_players[
        games_with_selected_players
    ].index.tolist()
    logging.info(f"Valid games: {valid_games}")

    # Filter DataFrame for valid games where the selected players started
    valid_games_data = team_data[team_data["game_id"].isin(valid_games)]
    logging.info(f"Valid games data: {valid_games_data.head()}")

    # logging statement that gives the total number of unique games with the selected players
    logging.info(f"Total number of unique games with the selected players: {len(valid_games)}")

    # Count how many times each player, not in selected or excluded players, started in these games
    most_common_starters = (
        valid_games_data.groupby("player").size().nlargest(6).reset_index(name='Starts Together')
    )
    most_common_starters.columns = ["Player", "Starts Together"]

    # Rest of the code remains the same...

    if set_piece_takers:
        # Set piece columns to calculate percentages
        set_piece_columns = [
            "Deadballs",
            "Freekicks",
            "Cornerkicks",
            "Inswinging",
            "Outswinging",
            "Straight",
        ]

        # Calculate team total set pieces for each game
        team_set_pieces = (
            valid_games_data.groupby("game_id")[set_piece_columns].sum().reset_index()
        )
        team_set_pieces["TotalSetPieces"] = team_set_pieces[set_piece_columns].sum(
            axis=1
        )

        # Merge team total back to the individual player data
        valid_games_data = valid_games_data.merge(
            team_set_pieces[["game_id", "TotalSetPieces"]], on="game_id"
        )

        # Calculate the sum of each type of set piece taken by each player per game
        player_set_pieces = (
            valid_games_data.groupby(["game_id", "player"])[set_piece_columns]
            .sum()
            .reset_index()
        )
        player_set_pieces = player_set_pieces.merge(
            team_set_pieces[["game_id", "TotalSetPieces"]], on="game_id"
        )

        # Calculate percentages for each player per game
        for column in set_piece_columns:
            player_set_pieces[f"{column}_Percent"] = (
                player_set_pieces[column] / player_set_pieces["TotalSetPieces"]
            ) * 100

        # Get the average percentage for each player across all games where the selected players started
        average_percentages = (
            player_set_pieces.groupby("player")[
                [f"{column}_Percent" for column in set_piece_columns]
            ]
            .mean()
            .reset_index()
        )

        # rename player to Player
        average_percentages = average_percentages.rename(columns={"player": "Player"})

        print(average_percentages.columns.tolist())

        # Merge the average percentages with most common starters
        most_common_starters = (
            valid_games_data["player"].value_counts().head(6).reset_index()
        )
        most_common_starters.columns = ["Player", "Starts Together"]
        most_common_starters = most_common_starters.merge(
            average_percentages, on="Player", how="left"
        )
    else:
        # Count how many times each player, not in selected or excluded players, started in these games
        most_common_starters = (
            valid_games_data["player"].value_counts().head(6).reset_index()
        )
        most_common_starters.columns = ["Player", "Starts Together"]

    # the players selected for analysis should not be included in the final most_common_starters DataFrame
    most_common_starters = most_common_starters[~most_common_starters["Player"].isin(selected_players)]

    # put the starts together as a percentage of the total games
    most_common_starters["Combo Freq"] = (
        most_common_starters["Starts Together"] / len(valid_games) * 100
    ).map("{:.0f}%".format)

    # put Starts Together then Starts Freq as the first two columns
    most_common_starters = most_common_starters[
        ["Player", "Starts Together", "Combo Freq"]
    ]

    # print logging statements that give the most common starters DataFrame and then lists off the players selected for analysis and the count of games they started together
    logging.info(f"Most common starters: {most_common_starters.head()}")
    logging.info(f"Players selected for analysis: {selected_players}")
    logging.info(f"Count of games they started together: {len(valid_games)}")

    # Prepare output text
    num_games = len(valid_games)
    players_joined = ", ".join(selected_players) if selected_players else "No players"
    excluded_joined = ", ".join(excluded_players) if excluded_players else "None"

    # Determine the correct grammar for selected players
    if len(selected_players) == 0:
        selected_text = "**Included player(s):** :red[None]\n"
    elif len(selected_players) == 1:
        selected_text = (
            f"**Included player(s):** ({len(selected_players)}) {selected_players[0]}\n"
        )
    else:
        selected_text = (
            f"**Included player(s):** ({len(selected_players)}) {players_joined}\n"
        )

    # Determine the correct grammar for excluded players
    if len(excluded_players) == 0:
        excluded_text = "**Excluded player(s):** :red[None]"
    elif len(excluded_players) == 1:
        excluded_text = (
            f"**Excluded player(s):** ({len(excluded_players)}) {excluded_players[0]}"
        )
    else:
        excluded_text = (
            f"**Excluded player(s):** ({len(excluded_players)}) {excluded_joined}"
        )

    # Modify the text based on the number of selected players and excluded players
    if len(selected_players) > 1 and len(excluded_players) > 0:
        text = f"Found {num_games} games where {players_joined} started together and {excluded_joined} did not start for {team_name}.\n"
    elif len(selected_players) == 1 and len(excluded_players) > 0:
        text = f"Found {num_games} games where {players_joined} started and {excluded_joined} did not start for {team_name}.\n"
    elif len(selected_players) > 1 and len(excluded_players) == 0:
        text = f"Found {num_games} games where {players_joined} started together for {team_name}.\n"
    elif len(selected_players) == 1 and len(excluded_players) == 0:
        text = (
            f"Found {num_games} games where {players_joined} started for {team_name}.\n"
        )
    else:  # case where there are no selected players but there are excluded players
        text = f"Found {num_games} games where {excluded_joined} did not start for {team_name}.\n"

    text += f"\n{selected_text}\n{excluded_text}"

    return most_common_starters, num_games, text

# create function to get anticorrelation between players, which should output a similar DataFrame to the one above
def get_anticorrelation_players(team_name, selected_players, excluded_players, dataframe):
    # Ensure selected_players and excluded_players are lists
    if not isinstance(selected_players, list):
        selected_players = [selected_players]
    if not isinstance(excluded_players, list):
        excluded_players = [excluded_players]

    print(
        f"Filtering for {team_name}. Including: {selected_players}. Excluding: {excluded_players}\n\n"
    )

    # Filter for is_starter == True and for the selected team
    dataframe = dataframe[dataframe["is_starter"] == True]
    team_data = dataframe[dataframe["team"] == team_name]

    # Function to filter games based on selected and excluded players
    def game_filter(players):
        return set(selected_players).issubset(set(players)) and set(
            excluded_players
        ).isdisjoint(set(players))

    # Apply the game filter
    games_with_selected_players = (
        team_data.groupby("game_id")["player"].apply(list).apply(game_filter)
    )
    valid_games = games_with_selected_players[
        games_with_selected_players
    ].index.tolist()

    # Filter DataFrame for valid games where the selected players started
    valid_games_data = team_data[team_data["game_id"].isin(valid_games)]

    # Count how many times each player, not in selected or excluded players, did not start in these games
    least_common_starters = (
        valid_games_data[~valid_games_data["player"].isin(selected_players + excluded_players)]["player"].value_counts().tail(6).reset_index()
    )
    least_common_starters.columns = ["Player", "Starts Apart"]
    
    # Prepare output text
    num_games = len(valid_games)
    players_joined = ", ".join(selected_players)
    excluded_joined = ", ".join(excluded_players) if excluded_players else "None"

    # Determine the correct grammar for selected players
    if len(selected_players) == 0:
        selected_text = "No players were selected"
    elif len(selected_players) == 1:
        selected_text = f"{selected_players[0]} was selected"
    else:
        selected_text = f"{len(selected_players)} players were selected"

    # Determine the correct grammar for excluded players
    if len(excluded_players) == 0:
        excluded_text = "No players were excluded"
    elif len(excluded_players) == 1:
        excluded_text = f"{excluded_players[0]} was excluded"
    else:
        excluded_text = f"{len(excluded_players)} players were excluded"

    # Modify the text based on the number of selected players
    if len(selected_players) > 1:
        text = f"Found {num_games} games where {players_joined} started together and {excluded_joined} did not start for {team_name}."
    else:
        text = f"Found {num_games} games where {players_joined} started and {excluded_joined} did not start for {team_name}."

    text += f" {selected_text} and {excluded_text}."

    return least_common_starters, num_games, text

# define a function to aggregate set piece takers
def main():
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # set config
    st.set_page_config(
        page_title="Football Lineup Analysis",
        page_icon="❗️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    socials_tag = "@DraftAlchemy"
    st.markdown(
        """
        <h1 style="font-size: 2.5em; text-align: left; color: aliceblue;">
            Football Lineup Analysis <span style="color: mistyrose;">by</span> <span style="color: wheat;">{}</span>
        </h1>
        """.format(
            socials_tag
        ),
        unsafe_allow_html=True,
    )

    # Load CSV file
    fbref_lineups = pd.read_csv("scraped_teamsheets/teamsheets_and_sets_20240421212817.csv")

    fbref_lineups.columns.tolist()

    # get all teams where the league is ENG-Premier League
    premier_league_teams = fbref_lineups[fbref_lineups["league"] == "ENG-Premier League"]["team"].unique()

    # Exclude goalkeepers and filter for 'ENG-Premier League' and starters only
    fbref_lineups = fbref_lineups[
        (fbref_lineups["position"] != "GK")
        # & (fbref_lineups["is_starter"] == True)
        & (fbref_lineups["team"].isin(premier_league_teams))
    ]

    # Add a 'game_id' column to uniquely identify each game
    fbref_lineups["game_id"] = (
        fbref_lineups["season"].astype(str) + ":" + fbref_lineups["game"]
    )

    logging.info(f"Unique opponents in fbref_lineups dataframe: {fbref_lineups['opponent'].unique()}")

    # Simplifying the league names and mapping seasons for display
    season_dict = {
        1617: "2016-2017",
        1718: "2017-2018",
        1819: "2018-2019",
        1920: "2019-2020",
        2021: "2020-2021",
        2122: "2021-2022",
        2223: "2022-2023",
        2324: "2023-2024",
    }
    fbref_lineups["season_display"] = fbref_lineups["season"].map(season_dict)
    fbref_lineups["league_display"] = fbref_lineups["league"].str.split("-").str[1]

    # add expander here to explain the app
    with st.expander("**:red[About this app]**", expanded=True):
        leagues = sorted(fbref_lineups["league_display"].unique().tolist())
        leagues_list = "\n".join([f'    - {league} ' for league in leagues])
        st.markdown(
            """
            The main function of this app is to analyze the team lineups and player positions in football matches.
            This app contains data from the following leagues:

            - Premier League
            - Champions League
            - Europa League
            - FA Cup
            - EFL Cup
            - Conference League

            This app uses the Apriori algorithm to analyze team lineups and player positions in football matches.
            You can select a season, team, and competition to view detailed player analysis and team profiles.
            """
        )

    set_piece_takers = False

    # create a toggle "Add Set Piece Data" which if clicked will filter to only include: seasons == [1718 1819 2021 2122 2223 2324] and leagues == ['ENG-Premier League' 'UEFA-Champions League' 'UEFA-Europa Conference League' 'UEFA-Europa League']
    if st.toggle("Add SetPiece Data", help="If this option is selected the data will be truncated as it filters for specific seasons"):
        set_piece_takers = True
        fbref_lineups = fbref_lineups[
            fbref_lineups["season"].isin([1718, 1819, 2021, 2122, 2223, 2324])
            & fbref_lineups["league"].isin(
                [
                    "ENG-Premier League",
                    "UEFA-Champions League",
                    "UEFA-Europa Conference League",
                    "UEFA-Europa League",
                ]
            )
        ]

    # Streamlit UI for season, team, and competition selection
    seasons = ["All Seasons"] + sorted(
        fbref_lineups["season_display"].unique().tolist(), reverse=True
    )
    teams = sorted(fbref_lineups["team"].unique().tolist())
    selected_season = st.selectbox("Select a season:", seasons)
    selected_team = st.selectbox("Select a team:", teams)

    # Filtering data based on user selection
    if selected_season != "All Seasons":
        filtered_data = fbref_lineups[
            (fbref_lineups["season_display"] == selected_season)
            & (fbref_lineups["team"] == selected_team)
        ]
    else:
        filtered_data = fbref_lineups[fbref_lineups["team"] == selected_team]

    comps = ["All Comps"] + sorted(filtered_data["league_display"].unique().tolist())
    selected_comp = st.selectbox("Select a competition:", comps)
    if selected_comp != "All Comps":
        filtered_data = filtered_data[filtered_data["league_display"] == selected_comp]

    # Sorting logic based on minutes played
    sort_by_minutes = st.checkbox("Sort by total minutes played", value=False)
    if sort_by_minutes:
        players = (
            filtered_data.groupby("player")["minutes_played"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        players = sorted(filtered_data["player"].unique().tolist())

    # Select players to exclude from analysis
    players_to_exclude = st.multiselect(":red[Exclude] player(s) from analysis", players, help="For example, you can exclude players who are not currently available...")

    # Create a copy of the original DataFrame
    fbref_lineups_copy = fbref_lineups.copy()

    # Exclude players from the copied DataFrame
    if players_to_exclude:
        for player in players_to_exclude:
            game_ids_to_exclude_based_on_player = filtered_data[
                filtered_data["player"] == player
            ]["game_id"].unique()
            filtered_data = filtered_data[
                ~filtered_data["game_id"].isin(game_ids_to_exclude_based_on_player)
            ]
            filtered_data = filtered_data[filtered_data["player"] != player]

    # Dynamically adjusting players for analysis based on exclusions
    players_for_analysis = [
        player for player in players if player not in players_to_exclude
    ]
    selected_players = st.multiselect(
        ":blue[Include] player(s) for analysis:", players_for_analysis
    )

    # Analyze button logic
    if st.button(f"Analyze"):
        tab1, tab2 = st.tabs(["🏃‍♂️ Players", "🔟 Team Profile"])

        # Ensuring there's a selection to analyze
        if not selected_players and not players_to_exclude:
            st.warning("Please select player(s) for analysis.")
            # Conduct general team specific analysis

        else:
            with tab1:
                col1, col2 = st.columns(2)

                st.title(f"Player Analysis for {selected_team}")

                with col1:
                    # Conduct analysis
                    most_common_players, _, text = get_most_common_players(
                        selected_team,
                        selected_players,
                        players_to_exclude,
                        filtered_data,
                        set_piece_takers=set_piece_takers,
                    )
                    st.write(text)
                    st.dataframe(most_common_players)
                
                # get anticorrelation players with col2
                with col2:
                    anti_corr_players, _, text = get_anticorrelation_players(
                        selected_team,
                        selected_players,
                        players_to_exclude,
                        filtered_data,
                    )
                    st.write(text)
                    st.dataframe(anti_corr_players)

                # Detailed player analysis for each selected player
                for player in selected_players:
                    positions, opponents, anti_opponents = get_player_positions_v2(
                        filtered_data, player, selected_team
                    )
                    st.write(f"Positions played by {player} under the above circumstances:")
                    st.dataframe(positions)
                    st.write(f"Opponents faced by {player} under the above circumstances:")
                    st.dataframe(opponents)
                    st.write(f"Opponents faced when {player} is not a starter:")
                    st.write(anti_opponents)

            with tab2:
                st.title(f"Team Profile Analysis for :rainbow[{selected_team}]")
                positions_data = get_positions_of_each_game(
                    filtered_data, selected_team
                )
                st.write(f"Positional setup by {selected_team}:")
                st.info(
                    f"'is_oop' is the average number of out-of-position players when {selected_team} uses the lineup. 'is_oop' is set as true if a starter is registered in a position that is not their most common position. 'count' is the number of games with the referenced positional setup."
                )
                st.dataframe(positions_data, use_container_width=True)
                # team_profile = get_team_profile(selected_team, filtered_data)
                # # reset the index for the team profile DataFrame
                # team_profile.reset_index(drop=True, inplace=True)
                # st.write(f"Team profile for {selected_team}:")
                # st.dataframe(team_profile)
                st.divider()
            # # Conduct analysis
            # most_common_players, _, text = get_most_common_players(
            #     selected_team,
            #     selected_players,
            #     players_to_exclude,
            #     fbref_lineups_copy,
            #     set_piece_takers=set_piece_takers,
            # )
            # st.write(text)
            # st.dataframe(most_common_players)

            # # Detailed player analysis for each selected player
            # for player in selected_players:
            #     positions, opponents = get_player_positions_v2(
            #         fbref_lineups_copy, player, selected_team
            #     )
            #     st.write(f"Positions played by {player} under the above circumstances:")
            #     st.dataframe(positions)
            #     st.write(f"Opponents faced by {player} under the above circumstances:")
            #     st.dataframe(opponents)
    # else:
    #     st.warning("Please select player(s) for analysis.")
    #     st.markdown(
    #         """
    #         <style>
    #         .reportview-container .markdown-text-container {
    #             font-family: monospace;
    #         }
    #         .sidebar .sidebar-content {
    #             background-image: linear-gradient(#2e7bcf,#2e7bcf);
    #             color: white;
    #         }
    #         .Widget>label {
    #             color: white;
    #             font-family: monospace;
    #         }
    #         [data-testid="stButton"] > div > div {
    #             background-color: #4CAF50;
    #             color: white;
    #         }
    #         </style>
    #         """,
    #         unsafe_allow_html=True,
    #     )
    #     if st.button(
    #         f":rainbow[Initiate Apriori algorithm to run Team Profile analysis for]: :white[{selected_team}]",
    #         use_container_width=True,
    #         type="secondary",
    #     ):
    #         positions_data = get_positions_of_each_game(filtered_data, selected_team)
    #         st.title(f"{selected_team}")
    #         st.write(f"Positional setup by {selected_team}:")
    #         st.info(
    #             f"'is_oop' is the average number of out-of-position players when {selected_team} uses the lineup. 'is_oop' is set as true if a starter is registered in a position that is not their most common position. 'count' is the number of games with the referenced positional setup."
    #         )
    #         st.dataframe(positions_data, use_container_width=True)
    #         # team_profile = get_team_profile(selected_team, filtered_data)
    #         # # reset the index for the team profile DataFrame
    #         # team_profile.reset_index(drop=True, inplace=True)
    #         # st.write(f"Team profile for {selected_team}:")
    #         # st.dataframe(team_profile)
    #         st.divider()


if __name__ == "__main__":
    main()
