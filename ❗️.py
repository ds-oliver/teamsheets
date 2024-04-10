from collections import Counter
import pandas as pd
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ignore FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


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

    # create a DataFrame to store the position counts, most recent date, and other players
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

    # iterate over each position and count the occurrences
    for position in positions:
        position_data = filtered_players[filtered_players["position"] == position]
        count = position_data.shape[0]
        most_recent_date = position_data["date"].max()
        other_players = (
            team_starters[team_starters["game"].isin(position_data["game"])]["player"]
            .unique()
            .tolist()
        )
        other_players.remove(player_match)
        home_games = position_data[position_data["home_team"] == team_name].shape[0]
        away_games = position_data[position_data["away_team"] == team_name].shape[0]
        position_counts = position_counts.append(
            {
                "Position": position,
                "Count": count,
                "Most Recent Date": most_recent_date,
                "Other Players": other_players,
                "Home Games": home_games,
                "Away Games": away_games,
            },
            ignore_index=True,
        )

    # sort the position counts by count in descending order
    position_counts = position_counts.sort_values(by="Count", ascending=False).reset_index(
        drop=True
    )

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


# def get_most_common_players(team_name, selected_players, dataframe):
#     """
#     Identifies the most common players who start games alongside the selected player(s) for the given team.

#     Parameters:
#     - team_name: The name of the team.
#     - selected_players: A list of player names to analyze.
#     - dataframe: The filtered DataFrame containing game lineup information.

#     Returns:
#     - A DataFrame with the most common players who started with the selected player(s), including the number of starts together.
#     - The total number of starts for the selected player(s).
#     - An informative text message summarizing the findings.
#     """

#     # Ensure selected_players is a list for consistent processing
#     if not isinstance(selected_players, list):
#         selected_players = [selected_players]

#     # Create a mask for games where each of the selected players started
#     mask = dataframe["player"].isin(selected_players) & (dataframe["team"] == team_name)
#     games_with_selected_players = dataframe[mask].groupby("game")["player"].agg(list)

#     # Filter games where all selected players started
#     games_where_all_started = games_with_selected_players[
#         games_with_selected_players.apply(
#             lambda players: all(player in players for player in selected_players)
#         )
#     ]

#     # List of all games IDs where all selected players have started
#     games_ids = games_where_all_started.index.tolist()

#     # Filter original dataframe for these games
#     filtered_games = dataframe[dataframe["game"].isin(games_ids)]

#     # Find other starters in these games
#     other_starters = filtered_games[~filtered_games["player"].isin(selected_players)][
#         "player"
#     ]

#     # Count the occurrences of each player
#     most_common_starters = other_starters.value_counts().head(6).reset_index()
#     most_common_starters.columns = ["Player", "Starts Together"]

#     # Prepare output text
#     players_joined = ", ".join(selected_players)
#     num_games = len(games_ids)
#     text = f"{num_games} games found where {players_joined} started together for {team_name}."

#     return most_common_starters, num_games, text

def get_most_common_players(team_name, selected_players, dataframe):
    # selected players is a list of player names
    if not isinstance(selected_players, list):
        selected_players = [selected_players]

    print(f"Filtering for {team_name} and {selected_players}\n\n")

    # create a mask for games where each of the selected players started
    mask = dataframe["player"].isin(selected_players) & (dataframe["team"] == team_name)
    # group the players by game
    games_with_selected_players = dataframe[mask].groupby("game_id")["player"].agg(list)

    # filter games where all selected players started
    games_where_all_started = games_with_selected_players[
        games_with_selected_players.apply(
            lambda players: all(player in players for player in selected_players)
        )
    ]

    # list of all games IDs where all selected players have started
    games_ids = games_where_all_started.index.tolist()

    # filter original dataframe for these games
    filtered_games = dataframe[dataframe["game_id"].isin(games_ids)]

    # find other starters in these games
    other_starters = filtered_games[~filtered_games["player"].isin(selected_players)][
        "player"
    ]

    # count the occurrences of each player
    most_common_starters = other_starters.value_counts().head(6).reset_index()
    most_common_starters.columns = ["Player", "Starts Together"]

    # prepare output text
    players_joined = ", ".join(selected_players)
    num_games = len(games_ids)
    
    # if we are only looking at one player we will return a different message
    if len(selected_players) == 1:
        text = f"{num_games} games found where {players_joined} started for {team_name}."
    else:
        text = f"{num_games} games found where {players_joined} started together for {team_name}."

    return most_common_starters, num_games, text

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
    positions_data = positions_data[positions_data["most_common_position"].apply(len) == 10]

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
        frequent_itemsets = apriori(positions_data_bool, min_support=min_support, use_colnames=True)
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
    team_starters = team_starters.rename(columns={"most_common_position": "starters_positions"})

    # group the players so we get position as a list, and sum the is_oop column
    positions_data = team_starters.groupby("game_id").agg(
        {"starters_positions": list, "is_oop": "sum"}
    )

    # reset index
    positions_data = positions_data.reset_index()

    # sort starters_positions according to the order list
    positions_data["starters_positions"] = positions_data["starters_positions"].apply(
        lambda x: sorted(x, key=lambda position: order.index(position) if position in order else len(order))
    )

    # convert list to tuple
    positions_data["starters_positions"] = positions_data["starters_positions"].apply(tuple)

    # group by starters_positions and aggregate is_oop and count
    positions_data = positions_data.groupby("starters_positions").agg(
        {"is_oop": "mean", "game_id": "count"}
    ).reset_index()

    # rename game_id to count
    positions_data = positions_data.rename(columns={"game_id": "count"})

    # make sure count is greater than 1
    positions_data = positions_data[positions_data["count"] > 1]

    # sort by count in descending order
    positions_data = positions_data.sort_values(by="count", ascending=False).reset_index(drop=True)

    return positions_data


def main():
    """
    The goal of this app is to analyze the most common players who started together for a team in a season. We will focus on the teams in the English Premier League but but across a number of seasons and competitions. We will also look at the positions a player has played and the opponents they faced. We will give the user the ability to filter by season, team, competition, and player(s). We will pass the selected player(s) to the get_most_common_players function to get the most common players who started together with the selected player(s). We will also pass the selected player to the get_player_positions function to get the positions the player has played and the opponents they faced. This means that the get_most_common_players function is designed to take multiple players from the same team and a dataframe as input, it will return the most common players who started together, the games where the selected players started together, the number of starts, and a text message. The get_player_positions function is designed to take a dataframe, a player name, and a team name as input, it will return the positions the player has played and the opponents they faced. We will display the most common players, the games where the selected players started together, the number of starts, and the text message. We will also display the positions the selected player has played and the opponents they faced. We will use the fbref_lineups DataFrame which contains the player lineups for the English Premier League teams across multiple seasons and competitions. The data we will pass to the functions will be player-specific and team-specific. We will use the Counter class from the collections module to count the occurrences of the players who started together. We will use the pandas library to manipulate the data and create DataFrames for display. We will use the streamlit library to create the web app and the selectbox and multiselect widgets for user input.
    """

    # load csv file
    fbref_lineups = pd.read_csv("/Users/hogan/soccerdata/fbref_lineups_epl_v5.csv")

    # filter out position == 'GK'
    fbref_lineups = fbref_lineups[fbref_lineups["position"] != "GK"]

    # Filter for 'ENG-Premier League' and starters only
    premier_league_teams = fbref_lineups[fbref_lineups["league"] == "ENG-Premier League"]["team"].unique()
    fbref_lineups = fbref_lineups[
        (fbref_lineups["is_starter"] == True)
        & (fbref_lineups["team"].isin(premier_league_teams))
    ]

    # Add a 'game_id' column to uniquely identify each game
    fbref_lineups["game_id"] = (
        fbref_lineups["season"].astype(str) + ":" + fbref_lineups["game"]
    )

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

    # put a toggle to sort the players list in the selectbox to show the players in order of total minutes_played or just alphabetically
    sort_by_minutes = st.checkbox("Sort by total minutes played", value=False)

    players_list_sorted_by_minutes = (
        filtered_data.groupby("player")["is_starter"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    
    players_list_sorted_alphabetically = filtered_data.sort_values(by="player")["player"].unique().tolist()

    # logic to sort the players list based on the checkbox
    if sort_by_minutes:
        players = players_list_sorted_by_minutes
    else:
        players = players_list_sorted_alphabetically

    selected_players = st.multiselect("Select player(s):", players)

    # Before calling the function, check if 'team' column exists in the DataFrame
    if 'team' not in filtered_data.columns:
        st.error("Data does not contain 'team' column.")
        st.error(f"Columns available: {filtered_data.columns.tolist()}")
    else:
        if selected_players:
            most_common_players, _, text = get_most_common_players(
                selected_team, selected_players, filtered_data
            )
            st.write(text)
            st.dataframe(most_common_players)

            # For detailed player analysis (assuming the get_player_positions function is correctly implemented)
            for player in selected_players:
                positions, opponents = get_player_positions(
                    filtered_data, player, selected_team
                )
                st.write(f"Positions played by {player}:")
                st.dataframe(positions)
                st.write(f"Opponents faced by {player}:")
                st.dataframe(opponents)
        else:
            # run team profile analysis
            # display button to run team profile analysis
            if st.button("Initiate Apriori algorithm to run Team Profile analysis"):
                positions_data = get_positions_of_each_game(filtered_data, selected_team)

                st.title(f"{selected_team}")

                st.write(f"Positional setup by {selected_team}:")
                st.dataframe(positions_data)
                # team_profile = get_team_profile(selected_team, filtered_data)
                # # reset the index for the team profile DataFrame
                # team_profile.reset_index(drop=True, inplace=True)

                # st.write(f"Team profile for {selected_team}:")
                # st.dataframe(team_profile)

if __name__ == "__main__":
    main()
