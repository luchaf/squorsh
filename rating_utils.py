import pandas as pd
from collections import defaultdict
from typing import Dict
from glicko2 import Player as Glicko2Player
import trueskill


def generate_elo_ratings(
    df_in: pd.DataFrame, base_elo: float = 1500, K: float = 20
) -> pd.DataFrame:
    """
    Given a DataFrame of matches (already filtered), compute Elo ratings for each player.
    """
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()
    elo_ratings: Dict[str, float] = defaultdict(lambda: base_elo)

    for _, row in df_sorted.iterrows():
        p1, p2 = row["Player1"], row["Player2"]
        r1, r2 = elo_ratings[p1], elo_ratings[p2]
        exp1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        exp2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))

        if row["Winner"] == p1:
            elo_ratings[p1] = r1 + K * (1.0 - exp1)
            elo_ratings[p2] = r2 + K * (0.0 - exp2)
        else:
            elo_ratings[p1] = r1 + K * (0.0 - exp1)
            elo_ratings[p2] = r2 + K * (1.0 - exp2)

    elo_df = pd.DataFrame(
        [(player, rating) for player, rating in elo_ratings.items()],
        columns=["Player", "Elo Rating"],
    )
    elo_df.sort_values("Elo Rating", ascending=False, inplace=True)
    elo_df.reset_index(drop=True, inplace=True)
    return elo_df


def generate_glicko2_ratings(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Glicko2 ratings for each player by iterating through matches in chronological order.
    Requires 'pip install glicko2'.
    """

    # Sort by date and match_number_total to ensure chronological processing
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()

    # Create a dictionary of Glicko2Player objects
    all_players = sorted(set(df_sorted["Player1"]) | set(df_sorted["Player2"]))
    players_dict = {p: Glicko2Player() for p in all_players}

    # Process each match in chronological order
    for _, row in df_sorted.iterrows():
        p1, p2 = row["Player1"], row["Player2"]
        player1 = players_dict[p1]
        player2 = players_dict[p2]

        # Current rating, rating deviation (RD), and volatility
        r1, rd1, vol1 = player1.rating, player1.rd, player1.vol
        r2, rd2, vol2 = player2.rating, player2.rd, player2.vol

        # Determine outcome from perspective of p1
        if row["Winner"] == p1:
            score_p1, score_p2 = 1.0, 0.0
        else:
            score_p1, score_p2 = 0.0, 1.0

        # Update p1's rating with p2 as the opponent
        player1.update_player([r2], [rd2], [score_p1])
        # Update p2's rating with p1 as the opponent
        player2.update_player([r1], [rd1], [score_p2])

    # Build output DataFrame
    results = []
    for p in all_players:
        pl = players_dict[p]
        results.append(
            {
                "Player": p,
                "Glicko2 Rating": pl.rating,
                "RD": pl.rd,
                "Volatility": pl.vol,
            }
        )
    df_glicko = pd.DataFrame(results)
    df_glicko.sort_values("Glicko2 Rating", ascending=False, inplace=True)
    df_glicko.reset_index(drop=True, inplace=True)
    return df_glicko


def generate_trueskill_ratings(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TrueSkill ratings by iterating matches in chronological order.
    By default, each player starts with Rating(mu=25, sigma=8.333...).
    """
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()

    # Dictionary of player -> trueskill.Rating()
    players_dict = {}
    all_players = sorted(set(df_sorted["Player1"]) | set(df_sorted["Player2"]))
    for p in all_players:
        players_dict[p] = trueskill.Rating()  # default mu=25, sigma=8.333

    for _, row in df_sorted.iterrows():
        p1, p2 = row["Player1"], row["Player2"]
        rating_p1 = players_dict[p1]
        rating_p2 = players_dict[p2]

        # If p1 is the winner, we do
        if row["Winner"] == p1:
            rating_p1, rating_p2 = trueskill.rate_1vs1(rating_p1, rating_p2)
        else:
            # p2 is the winner, so invert the order
            rating_p2, rating_p1 = trueskill.rate_1vs1(rating_p2, rating_p1)

        players_dict[p1] = rating_p1
        players_dict[p2] = rating_p2

    data_out = []
    for p in all_players:
        r = players_dict[p]
        data_out.append(
            {
                "Player": p,
                "TrueSkill Mu": r.mu,
                "TrueSkill Sigma": r.sigma,
                "TrueSkill Rating": r.mu - 3 * r.sigma,  # conservative rating
            }
        )
    df_ts = pd.DataFrame(data_out)
    # Sort by the conservative skill estimate
    df_ts.sort_values("TrueSkill Rating", ascending=False, inplace=True)
    df_ts.reset_index(drop=True, inplace=True)
    return df_ts
