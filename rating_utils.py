import pandas as pd
import altair as alt
import streamlit as st
from collections import defaultdict
from glicko2 import Player as Glicko2Player
import trueskill


def generate_elo_ratings_over_time(
    df_in: pd.DataFrame, base_elo: float = 1500, K: float = 20
) -> pd.DataFrame:
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()
    elo_ratings = defaultdict(lambda: base_elo)
    rating_history = []

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

        rating_history.extend(
            [
                {"date": row["date"], "Player": p1, "Elo Rating": elo_ratings[p1]},
                {"date": row["date"], "Player": p2, "Elo Rating": elo_ratings[p2]},
            ]
        )

    return pd.DataFrame(rating_history)


def generate_glicko2_ratings_over_time(df_in: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()
    players = {
        p: Glicko2Player()
        for p in set(df_sorted["Player1"]) | set(df_sorted["Player2"])
    }
    rating_history = []

    for _, row in df_sorted.iterrows():
        p1, p2 = row["Player1"], row["Player2"]
        player1, player2 = players[p1], players[p2]

        score_p1 = 1.0 if row["Winner"] == p1 else 0.0
        score_p2 = 1.0 - score_p1

        player1.update_player([player2.rating], [player2.rd], [score_p1])
        player2.update_player([player1.rating], [player1.rd], [score_p2])

        rating_history.extend(
            [
                {
                    "date": row["date"],
                    "Player": p1,
                    "Glicko2 Rating": player1.rating,
                    "RD": player1.rd,
                    "Volatility": player1.vol,
                },
                {
                    "date": row["date"],
                    "Player": p2,
                    "Glicko2 Rating": player2.rating,
                    "RD": player2.rd,
                    "Volatility": player2.vol,
                },
            ]
        )

    return pd.DataFrame(rating_history)


def generate_trueskill_ratings_over_time(df_in: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df_in.sort_values(["date", "match_number_total"]).copy()
    players = {
        p: trueskill.Rating()
        for p in set(df_sorted["Player1"]) | set(df_sorted["Player2"])
    }
    rating_history = []

    for _, row in df_sorted.iterrows():
        p1, p2 = row["Player1"], row["Player2"]
        r1, r2 = players[p1], players[p2]

        if row["Winner"] == p1:
            r1, r2 = trueskill.rate_1vs1(r1, r2)
        else:
            r2, r1 = trueskill.rate_1vs1(r2, r1)

        players[p1], players[p2] = r1, r2

        rating_history.extend(
            [
                {
                    "date": row["date"],
                    "Player": p1,
                    "TrueSkill Rating": r1.mu - 3 * r1.sigma,
                    "Mu": r1.mu,
                    "Sigma": r1.sigma,
                },
                {
                    "date": row["date"],
                    "Player": p2,
                    "TrueSkill Rating": r2.mu - 3 * r2.sigma,
                    "Mu": r2.mu,
                    "Sigma": r2.sigma,
                },
            ]
        )

    return pd.DataFrame(rating_history)


def plot_ratings_over_time(
    df: pd.DataFrame, rating_column: str, title: str
) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{rating_column}:Q", title=rating_column),
            color=alt.Color("Player:N", title="Player"),
            tooltip=["date:T", "Player:N", f"{rating_column}:Q"],
        )
        .properties(title=title, width=800, height=400)
    )
