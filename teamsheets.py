from collections import Counter
import pandas as pd
import streamlit as st
import warnings

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


def get_most_common_players(team_name, selected_players, excluded_players, dataframe):
    if not isinstance(selected_players, list):
        selected_players = [selected_players]
    if not isinstance(excluded_players, list):
        excluded_players = [excluded_players]

    print(
        f"Filtering for {team_name}. Including: {selected_players}. Excluding: {excluded_players}\n\n"
    )

    # Filter for the selected team
    team_data = dataframe[dataframe["team"] == team_name]

    # Create a mask for games where any of the selected players started and none of the excluded players did
    def game_filter(players):
        return set(selected_players).issubset(set(players)) and set(
            excluded_players
        ).isdisjoint(set(players))

    # Group games by 'game_id' and filter using the mask
    games_with_selected_players = (
        team_data.groupby("game_id")["player"].apply(list).apply(game_filter)
    )
    valid_games = games_with_selected_players[
        games_with_selected_players
    ].index.tolist()

    # Filter the DataFrame for the valid games
    valid_games_data = team_data[team_data["game_id"].isin(valid_games)]

    # Count how many times each player, not in selected or excluded players, started in these games
    other_starters = valid_games_data[
        ~valid_games_data["player"].isin(selected_players + excluded_players)
    ]
    most_common_starters = other_starters["player"].value_counts().head(6).reset_index()
    most_common_starters.columns = ["Player", "Starts Together"]

    # Prepare output text
    num_games = len(valid_games)
    players_joined = ", ".join(selected_players)
    excluded_joined = ", ".join(excluded_players) if excluded_players else "None"
    text = f"Found {num_games} games where {players_joined} started together and {excluded_joined} did not start for {team_name}."

    return most_common_starters, num_games, text


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


def main():

    # add expander here to explain the app
    with st.expander("**About this app**"):
        st.markdown(
            """
            The main function of this app is to analyze the team lineups and player positions in football matches.
            This app uses the Apriori algorithm to analyze team lineups and player positions in football matches.
            You can select a season, team, and competition to view detailed player analysis and team profiles.
            """
        )
    # Load CSV file
    fbref_lineups = pd.read_csv("fbref_lineups_epl_v5.csv")

    # Exclude goalkeepers and filter for 'ENG-Premier League' and starters only
    fbref_lineups = fbref_lineups[
        (fbref_lineups["position"] != "GK")
        & (fbref_lineups["league"] == "ENG-Premier League")
        & (fbref_lineups["is_starter"] == True)
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
    st.subheader("Exclude Players")
    st.info("For example, you can exclude players who are injured")
    players_to_exclude = st.multiselect("Select player(s) to :red[Exclude]:", players)

    # Create a copy of the original DataFrame
    fbref_lineups_copy = fbref_lineups.copy()

    # Exclude players from the copied DataFrame
    for player in players_to_exclude:
        fbref_lineups_copy = fbref_lineups_copy[fbref_lineups_copy["player"] != player]

    # Select players for analysis, ensuring we exclude any players selected for exclusion
    st.subheader("Select Players for Analysis")
    selected_players = st.multiselect(
        "Select player(s) to :green[Group]:",
        [player for player in players if player not in players_to_exclude],
    )

    # Use the copied DataFrame for analysis
    if selected_players and players_to_exclude:
        # Exclude players from the copied DataFrame
        for player in players_to_exclude:
            fbref_lineups_copy = fbref_lineups_copy[
                fbref_lineups_copy["player"] != player
            ]

        most_common_players, _, text = get_most_common_players(
            selected_team, selected_players, players_to_exclude, fbref_lineups_copy
        )
        st.write(text)
        st.dataframe(most_common_players)
        # Detailed player analysis for each selected player
        for player in selected_players:
            positions, opponents = get_player_positions(
                fbref_lineups, player, selected_team
            )
            st.write(f"Positions played by {player}:")
            st.dataframe(positions)
            st.write(f"Opponents faced by {player}:")
            st.dataframe(opponents)
    elif players_to_exclude:
        most_common_players, _, text = get_most_common_players(
            selected_team, selected_players, players_to_exclude, fbref_lineups
        )
        st.write(text)
        st.dataframe(most_common_players)
        # Detailed player analysis for each selected player
        for player in selected_players:
            positions, opponents = get_player_positions(
                fbref_lineups, player, selected_team
            )
            st.write(f"Positions played by {player}:")
            st.dataframe(positions)
            st.write(f"Opponents faced by {player}:")
            st.dataframe(opponents)
    else:
        st.warning("Please select player(s) for analysis.")
        st.markdown(
            """
            <style>
            .reportview-container .markdown-text-container {
                font-family: monospace;
            }
            .sidebar .sidebar-content {
                background-image: linear-gradient(#2e7bcf,#2e7bcf);
                color: white;
            }
            .Widget>label {
                color: white;
                font-family: monospace;
            }
            [data-testid="stButton"] > div > div {
                background-color: #4CAF50;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f":rainbow[Initiate Apriori algorithm to run Team Profile analysis for]: :white[{selected_team}]",
            use_container_width=True,
            type="secondary",
        ):
            positions_data = get_positions_of_each_game(filtered_data, selected_team)
            st.title(f"{selected_team}")
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


if __name__ == "__main__":
    main()
