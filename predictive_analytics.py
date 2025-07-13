import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict
import streamlit as st
from scipy import stats
from datetime import datetime, timedelta
import math


def calculate_head_to_head_probability(
    df: pd.DataFrame, 
    player1: str, 
    player2: str,
    elo_ratings: Dict[str, float] = None
) -> Dict[str, float]:
    """Calculate win probability based on historical matchups and ratings"""
    
    # Historical H2H data
    h2h_matches = df[
        ((df["Player1"] == player1) & (df["Player2"] == player2)) |
        ((df["Player1"] == player2) & (df["Player2"] == player1))
    ]
    
    if len(h2h_matches) > 0:
        p1_wins = len(h2h_matches[h2h_matches["Winner"] == player1])
        p2_wins = len(h2h_matches[h2h_matches["Winner"] == player2])
        total_h2h = p1_wins + p2_wins
        
        # Historical probability
        h2h_prob_p1 = p1_wins / total_h2h if total_h2h > 0 else 0.5
        
        # Recent form weight (last 3 matches)
        recent_h2h = h2h_matches.tail(3)
        recent_p1_wins = len(recent_h2h[recent_h2h["Winner"] == player1])
        recent_weight = 0.3 if len(recent_h2h) >= 3 else 0.1
        recent_prob_p1 = recent_p1_wins / len(recent_h2h) if len(recent_h2h) > 0 else 0.5
    else:
        h2h_prob_p1 = 0.5
        recent_prob_p1 = 0.5
        recent_weight = 0
    
    # Elo-based probability
    if elo_ratings and player1 in elo_ratings and player2 in elo_ratings:
        rating_diff = elo_ratings[player1] - elo_ratings[player2]
        elo_prob_p1 = 1 / (1 + 10**(-rating_diff/400))
        elo_weight = 0.5
    else:
        elo_prob_p1 = 0.5
        elo_weight = 0
    
    # Current form probability (last 10 matches overall)
    p1_recent = df[
        (df["Player1"] == player1) | (df["Player2"] == player2)
    ].tail(10)
    p2_recent = df[
        (df["Player1"] == player2) | (df["Player2"] == player2)
    ].tail(10)
    
    p1_form = len(p1_recent[p1_recent["Winner"] == player1]) / len(p1_recent) if len(p1_recent) > 0 else 0.5
    p2_form = len(p2_recent[p2_recent["Winner"] == player2]) / len(p2_recent) if len(p2_recent) > 0 else 0.5
    
    # Normalize form
    total_form = p1_form + p2_form
    if total_form > 0:
        form_prob_p1 = p1_form / total_form
    else:
        form_prob_p1 = 0.5
    
    # Weighted combination
    historical_weight = 0.3 - recent_weight
    form_weight = 0.2
    
    # Ensure weights sum to 1
    total_weight = historical_weight + recent_weight + elo_weight + form_weight
    if total_weight > 0:
        final_prob_p1 = (
            (h2h_prob_p1 * historical_weight +
             recent_prob_p1 * recent_weight +
             elo_prob_p1 * elo_weight +
             form_prob_p1 * form_weight) / total_weight
        )
    else:
        final_prob_p1 = 0.5
    
    # Calculate confidence based on sample size
    confidence = min(100, (len(h2h_matches) * 5 + len(p1_recent) + len(p2_recent)) * 2)
    
    return {
        "player1": player1,
        "player2": player2,
        "player1_probability": final_prob_p1,
        "player2_probability": 1 - final_prob_p1,
        "confidence": confidence,
        "h2h_matches": len(h2h_matches),
        "factors": {
            "historical": h2h_prob_p1,
            "recent_h2h": recent_prob_p1,
            "elo": elo_prob_p1,
            "current_form": form_prob_p1
        }
    }


def predict_score_distribution(
    df: pd.DataFrame,
    player1: str,
    player2: str
) -> Dict[str, any]:
    """Predict likely score distribution for a match"""
    
    # Get historical scores for both players
    p1_scores = []
    p2_scores = []
    
    for _, match in df.iterrows():
        if match["Player1"] == player1:
            p1_scores.append(match["Score1"])
        elif match["Player2"] == player1:
            p1_scores.append(match["Score2"])
        
        if match["Player1"] == player2:
            p2_scores.append(match["Score1"])
        elif match["Player2"] == player2:
            p2_scores.append(match["Score2"])
    
    # H2H specific scores
    h2h_matches = df[
        ((df["Player1"] == player1) & (df["Player2"] == player2)) |
        ((df["Player1"] == player2) & (df["Player2"] == player1))
    ]
    
    h2h_p1_scores = []
    h2h_p2_scores = []
    
    for _, match in h2h_matches.iterrows():
        if match["Player1"] == player1:
            h2h_p1_scores.append(match["Score1"])
            h2h_p2_scores.append(match["Score2"])
        else:
            h2h_p1_scores.append(match["Score2"])
            h2h_p2_scores.append(match["Score1"])
    
    # Calculate expected scores
    if h2h_p1_scores and h2h_p2_scores:
        # Weight H2H heavily if available
        expected_p1 = np.mean(h2h_p1_scores) * 0.7 + np.mean(p1_scores) * 0.3
        expected_p2 = np.mean(h2h_p2_scores) * 0.7 + np.mean(p2_scores) * 0.3
        std_p1 = np.std(h2h_p1_scores) * 0.7 + np.std(p1_scores) * 0.3
        std_p2 = np.std(h2h_p2_scores) * 0.7 + np.std(p2_scores) * 0.3
    else:
        expected_p1 = np.mean(p1_scores) if p1_scores else 8
        expected_p2 = np.mean(p2_scores) if p2_scores else 8
        std_p1 = np.std(p1_scores) if len(p1_scores) > 1 else 3
        std_p2 = np.std(p2_scores) if len(p2_scores) > 1 else 3
    
    # Generate score probabilities
    score_probs = {}
    
    # Common squash scores - only valid game-ending scores
    for p1_score in range(0, 16):
        for p2_score in range(0, 16):
            # Valid squash game-ending score check
            is_valid_score = False
            
            # Standard win: reach 11 with opponent under 10
            if (p1_score == 11 and p2_score < 10) or (p2_score == 11 and p1_score < 10):
                is_valid_score = True
            
            # Deuce situation: must win by 2 after 10-10
            elif p1_score >= 10 and p2_score >= 10:
                if abs(p1_score - p2_score) == 2:
                    is_valid_score = True
            
            if is_valid_score:
                # Calculate probability using normal distribution
                p1_prob = stats.norm.pdf(p1_score, expected_p1, std_p1)
                p2_prob = stats.norm.pdf(p2_score, expected_p2, std_p2)
                
                # Combined probability
                score_prob = p1_prob * p2_prob
                
                # Boost common scores
                if (p1_score == 11 and p2_score <= 9) or (p2_score == 11 and p1_score <= 9):
                    score_prob *= 1.5  # More likely to end before deuce
                
                score_probs[f"{p1_score}-{p2_score}"] = score_prob
    
    # Normalize probabilities
    total_prob = sum(score_probs.values())
    if total_prob > 0:
        score_probs = {k: v/total_prob for k, v in score_probs.items()}
    
    # Find most likely scores
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    top_scores = sorted_scores[:5]
    
    # Generate a valid expected score based on averages
    exp_p1_int = int(round(expected_p1))
    exp_p2_int = int(round(expected_p2))
    
    # Ensure it's a valid squash score
    if exp_p1_int > exp_p2_int:
        if exp_p1_int < 11:
            exp_p1_int = 11
        if exp_p1_int == 11 and exp_p2_int >= 10:
            exp_p2_int = 9
        elif exp_p1_int > 11 and exp_p1_int - exp_p2_int < 2:
            exp_p2_int = exp_p1_int - 2
    else:
        if exp_p2_int < 11:
            exp_p2_int = 11
        if exp_p2_int == 11 and exp_p1_int >= 10:
            exp_p1_int = 9
        elif exp_p2_int > 11 and exp_p2_int - exp_p1_int < 2:
            exp_p1_int = exp_p2_int - 2
    
    # Use most likely score if available
    if top_scores:
        expected_score = top_scores[0][0]
    else:
        expected_score = f"{exp_p1_int}-{exp_p2_int}"
    
    return {
        "expected_score": expected_score,
        "player1_avg": expected_p1,
        "player2_avg": expected_p2,
        "top_predictions": top_scores,
        "all_probabilities": score_probs
    }


def calculate_upset_potential(
    df: pd.DataFrame,
    elo_ratings: Dict[str, float] = None
) -> pd.DataFrame:
    """Calculate upset potential for upcoming matches"""
    
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    upset_data = []
    
    for i, p1 in enumerate(players):
        for p2 in players[i+1:]:
            # Get win probability
            prob_data = calculate_head_to_head_probability(df, p1, p2, elo_ratings)
            
            # Determine favorite and underdog
            if prob_data["player1_probability"] > 0.5:
                favorite = p1
                underdog = p2
                favorite_prob = prob_data["player1_probability"]
            else:
                favorite = p2
                underdog = p1
                favorite_prob = prob_data["player2_probability"]
            
            upset_prob = 1 - favorite_prob
            
            # Calculate upset factors
            # 1. Underdog recent form
            underdog_recent = df[
                (df["Player1"] == underdog) | (df["Player2"] == underdog)
            ].tail(5)
            underdog_form = len(underdog_recent[underdog_recent["Winner"] == underdog]) / len(underdog_recent) \
                if len(underdog_recent) > 0 else 0
            
            # 2. Favorite recent struggles
            favorite_recent = df[
                (df["Player1"] == favorite) | (df["Player2"] == favorite)
            ].tail(5)
            favorite_struggles = 1 - (len(favorite_recent[favorite_recent["Winner"] == favorite]) / len(favorite_recent)) \
                if len(favorite_recent) > 0 else 0
            
            # 3. Historical upsets between these players
            h2h = df[
                ((df["Player1"] == p1) & (df["Player2"] == p2)) |
                ((df["Player1"] == p2) & (df["Player2"] == p1))
            ]
            
            if len(h2h) > 0 and elo_ratings:
                historical_upsets = 0
                for _, match in h2h.iterrows():
                    match_date = match["date"]
                    # Would need historical ratings for accurate calculation
                    # Simplified: if underdog won, count as upset
                    if match["Winner"] == underdog:
                        historical_upsets += 1
                upset_history = historical_upsets / len(h2h)
            else:
                upset_history = 0
            
            # Calculate upset index
            upset_index = (upset_prob * 40 + 
                          underdog_form * 30 + 
                          favorite_struggles * 20 + 
                          upset_history * 10)
            
            if upset_index > 25:  # Threshold for "upset alert"
                upset_data.append({
                    "Favorite": favorite,
                    "Underdog": underdog,
                    "Favorite Win %": favorite_prob * 100,
                    "Upset Probability": upset_prob * 100,
                    "Underdog Form": underdog_form,
                    "Favorite Struggles": favorite_struggles,
                    "Upset Index": upset_index,
                    "Alert Level": "🔴 HIGH" if upset_index > 40 else "🟡 MEDIUM"
                })
    
    return pd.DataFrame(upset_data).sort_values("Upset Index", ascending=False)


def simulate_tournament_bracket(
    df: pd.DataFrame,
    players: List[str],
    elo_ratings: Dict[str, float] = None,
    num_simulations: int = 1000
) -> Dict[str, any]:
    """Simulate tournament outcomes using Monte Carlo method"""
    
    if len(players) not in [4, 8, 16]:
        return {"error": "Tournament size must be 4, 8, or 16 players"}
    
    tournament_wins = defaultdict(int)
    final_appearances = defaultdict(int)
    matchup_results = defaultdict(lambda: defaultdict(int))
    
    for sim in range(num_simulations):
        remaining_players = players.copy()
        round_num = 1
        
        while len(remaining_players) > 1:
            next_round = []
            
            # Pair up players
            for i in range(0, len(remaining_players), 2):
                p1 = remaining_players[i]
                p2 = remaining_players[i + 1]
                
                # Get win probability
                prob_data = calculate_head_to_head_probability(df, p1, p2, elo_ratings)
                
                # Simulate match
                if np.random.random() < prob_data["player1_probability"]:
                    winner = p1
                    loser = p2
                else:
                    winner = p2
                    loser = p1
                
                next_round.append(winner)
                matchup_results[winner][loser] += 1
                
                # Track finals
                if len(remaining_players) == 2:
                    final_appearances[p1] += 1
                    final_appearances[p2] += 1
                    tournament_wins[winner] += 1
            
            remaining_players = next_round
            round_num += 1
    
    # Calculate probabilities
    results = []
    for player in players:
        results.append({
            "Player": player,
            "Win %": (tournament_wins[player] / num_simulations) * 100,
            "Final %": (final_appearances[player] / num_simulations) * 100,
            "Avg Wins": sum(1 for opp in matchup_results[player]) / num_simulations
        })
    
    results_df = pd.DataFrame(results).sort_values("Win %", ascending=False)
    
    # Most common final
    finals_counter = defaultdict(int)
    for sim in range(100):  # Quick re-run for finals
        remaining = players.copy()
        while len(remaining) > 2:
            next_round = []
            for i in range(0, len(remaining), 2):
                prob_data = calculate_head_to_head_probability(df, remaining[i], remaining[i+1], elo_ratings)
                if np.random.random() < prob_data["player1_probability"]:
                    next_round.append(remaining[i])
                else:
                    next_round.append(remaining[i+1])
            remaining = next_round
        
        final_key = tuple(sorted(remaining))
        finals_counter[final_key] += 1
    
    most_likely_final = max(finals_counter.items(), key=lambda x: x[1])
    
    return {
        "predictions": results_df,
        "most_likely_final": most_likely_final[0],
        "final_probability": most_likely_final[1],
        "simulations": num_simulations
    }


def calculate_performance_trajectory(
    df: pd.DataFrame,
    player: str,
    days_ahead: int = 30
) -> Dict[str, any]:
    """Project future performance trajectory based on trends"""
    
    player_matches = df[
        (df["Player1"] == player) | (df["Player2"] == player)
    ].sort_values("date")
    
    if len(player_matches) < 10:
        return {"error": "Insufficient data for projection"}
    
    # Calculate rolling win rate
    win_rates = []
    dates = []
    
    for i in range(10, len(player_matches) + 1):
        window = player_matches.iloc[i-10:i]
        wins = len(window[window["Winner"] == player])
        win_rate = wins / len(window)
        win_rates.append(win_rate)
        dates.append(window.iloc[-1]["date"])
    
    # Fit trend line
    x = np.arange(len(win_rates))
    y = np.array(win_rates)
    
    # Try polynomial fit for non-linear trends
    if len(x) > 20:
        poly_degree = 2
        coeffs = np.polyfit(x, y, poly_degree)
        poly_fn = np.poly1d(coeffs)
        
        # Project forward
        future_x = np.arange(len(x), len(x) + days_ahead)
        future_y = poly_fn(future_x)
        
        # Constrain to [0, 1]
        future_y = np.clip(future_y, 0, 1)
    else:
        # Simple linear projection
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(x), len(x) + days_ahead)
        future_y = slope * future_x + intercept
        future_y = np.clip(future_y, 0, 1)
    
    # Calculate confidence intervals
    residuals = y - np.polyval(coeffs if len(x) > 20 else [slope, intercept], x)
    std_error = np.std(residuals)
    
    # Performance zones
    current_rate = win_rates[-1]
    projected_rate = future_y[-1]
    
    if projected_rate > current_rate + 0.1:
        trajectory = "📈 Strong Improvement"
    elif projected_rate > current_rate + 0.05:
        trajectory = "📊 Moderate Improvement"
    elif projected_rate < current_rate - 0.1:
        trajectory = "📉 Concerning Decline"
    elif projected_rate < current_rate - 0.05:
        trajectory = "📊 Slight Decline"
    else:
        trajectory = "➡️ Stable"
    
    return {
        "current_win_rate": current_rate,
        "projected_win_rate": projected_rate,
        "trajectory": trajectory,
        "confidence_interval": (
            max(0, projected_rate - 2 * std_error),
            min(1, projected_rate + 2 * std_error)
        ),
        "days_projected": days_ahead,
        "trend_strength": abs(projected_rate - current_rate)
    }


def what_if_scenario_analysis(
    df: pd.DataFrame,
    player: str,
    scenario_wins: int,
    scenario_losses: int,
    elo_ratings: Dict[str, float] = None
) -> Dict[str, any]:
    """Analyze impact of hypothetical match results"""
    
    # Current stats
    player_matches = df[(df["Player1"] == player) | (df["Player2"] == player)]
    current_wins = len(player_matches[player_matches["Winner"] == player])
    current_total = len(player_matches)
    current_rate = current_wins / current_total if current_total > 0 else 0
    
    # Projected stats
    new_wins = current_wins + scenario_wins
    new_total = current_total + scenario_wins + scenario_losses
    new_rate = new_wins / new_total if new_total > 0 else 0
    
    # Rating impact (simplified)
    if elo_ratings and player in elo_ratings:
        current_rating = elo_ratings[player]
        # Assume average opponent
        avg_opponent_rating = np.mean(list(elo_ratings.values()))
        
        # Calculate rating change
        K = 20
        rating_change = 0
        
        for _ in range(scenario_wins):
            expected = 1 / (1 + 10**((avg_opponent_rating - current_rating) / 400))
            rating_change += K * (1 - expected)
        
        for _ in range(scenario_losses):
            expected = 1 / (1 + 10**((avg_opponent_rating - current_rating) / 400))
            rating_change += K * (0 - expected)
        
        new_rating = current_rating + rating_change
    else:
        current_rating = 1500
        new_rating = 1500
        rating_change = 0
    
    # Rank impact
    if elo_ratings:
        current_rank = sorted(elo_ratings.values(), reverse=True).index(current_rating) + 1
        all_ratings = list(elo_ratings.values())
        all_ratings.remove(current_rating)
        all_ratings.append(new_rating)
        all_ratings.sort(reverse=True)
        new_rank = all_ratings.index(new_rating) + 1
    else:
        current_rank = 0
        new_rank = 0
    
    return {
        "current_record": f"{current_wins}-{current_total - current_wins}",
        "new_record": f"{new_wins}-{new_total - new_wins}",
        "current_win_rate": current_rate,
        "new_win_rate": new_rate,
        "win_rate_change": new_rate - current_rate,
        "current_rating": current_rating,
        "new_rating": new_rating,
        "rating_change": rating_change,
        "current_rank": current_rank,
        "new_rank": new_rank,
        "rank_change": current_rank - new_rank
    }