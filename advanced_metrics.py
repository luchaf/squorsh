import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import streamlit as st
from scipy import stats
from datetime import timedelta


def calculate_clutch_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate clutch performance metrics for close matches (9-11, 10-12, etc.)"""
    clutch_matches = df[
        (
            ((df["Score1"] >= 9) & (df["Score2"] == 11)) |
            ((df["Score2"] >= 9) & (df["Score1"] == 11)) |
            ((df["Score1"] >= 10) & (df["Score2"] >= 12)) |
            ((df["Score2"] >= 10) & (df["Score1"] >= 12))
        )
    ].copy()
    
    if clutch_matches.empty:
        return pd.DataFrame()
    
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    clutch_stats = []
    
    for player in players:
        player_clutch = clutch_matches[
            (clutch_matches["Player1"] == player) | (clutch_matches["Player2"] == player)
        ]
        
        clutch_wins = len(player_clutch[player_clutch["Winner"] == player])
        clutch_total = len(player_clutch)
        clutch_rate = clutch_wins / clutch_total if clutch_total > 0 else 0
        
        # Compare to overall win rate
        overall_matches = df[(df["Player1"] == player) | (df["Player2"] == player)]
        overall_wins = len(overall_matches[overall_matches["Winner"] == player])
        overall_rate = overall_wins / len(overall_matches) if len(overall_matches) > 0 else 0
        
        clutch_differential = clutch_rate - overall_rate
        
        clutch_stats.append({
            "Player": player,
            "Clutch Matches": clutch_total,
            "Clutch Wins": clutch_wins,
            "Clutch Win Rate": clutch_rate,
            "Overall Win Rate": overall_rate,
            "Clutch Differential": clutch_differential,
            "Clutch Rating": clutch_rate * np.sqrt(clutch_total)  # Weight by sample size
        })
    
    return pd.DataFrame(clutch_stats)


def analyze_comebacks(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze comeback victories - games won after trailing significantly"""
    comeback_threshold = 5  # Points behind
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    comeback_stats = []
    
    for player in players:
        player_matches = df[
            (df["Player1"] == player) | (df["Player2"] == player)
        ].copy()
        
        comebacks = 0
        comeback_opportunities = 0
        biggest_comeback = 0
        
        for _, match in player_matches.iterrows():
            if match["Player1"] == player:
                player_score = match["Score1"]
                opponent_score = match["Score2"]
            else:
                player_score = match["Score2"]
                opponent_score = match["Score1"]
            
            # Estimate if player was trailing (simplified - would need point-by-point data for accuracy)
            if match["Winner"] == player and opponent_score >= 7:
                # Potential comeback scenario
                deficit = opponent_score - (player_score - comeback_threshold)
                if deficit > 0:
                    comebacks += 1
                    biggest_comeback = max(biggest_comeback, deficit)
            
            if opponent_score >= player_score + comeback_threshold:
                comeback_opportunities += 1
        
        comeback_stats.append({
            "Player": player,
            "Comebacks": comebacks,
            "Comeback Opportunities": comeback_opportunities,
            "Comeback Rate": comebacks / comeback_opportunities if comeback_opportunities > 0 else 0,
            "Biggest Comeback": biggest_comeback,
            "Mental Toughness Score": comebacks * 10 + biggest_comeback * 5
        })
    
    return pd.DataFrame(comeback_stats)


def calculate_dominance_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dominance score combining win rate and average margin"""
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    dominance_stats = []
    
    for player in players:
        # Win rate calculation
        player_matches = df[(df["Player1"] == player) | (df["Player2"] == player)]
        wins = len(player_matches[player_matches["Winner"] == player])
        total = len(player_matches)
        win_rate = wins / total if total > 0 else 0
        
        # Average margin calculation
        player_wins = df[df["Winner"] == player]
        avg_victory_margin = player_wins["PointDiff"].mean() if len(player_wins) > 0 else 0
        
        player_losses = df[df["Loser"] == player]
        avg_defeat_margin = abs(player_losses["LoserPointDiff"].mean()) if len(player_losses) > 0 else 0
        
        # Dominance score formula
        dominance = (win_rate * 100) + (avg_victory_margin * 5) - (avg_defeat_margin * 2)
        
        dominance_stats.append({
            "Player": player,
            "Win Rate": win_rate,
            "Avg Victory Margin": avg_victory_margin,
            "Avg Defeat Margin": avg_defeat_margin,
            "Dominance Score": dominance,
            "Total Matches": total
        })
    
    return pd.DataFrame(dominance_stats)


def calculate_consistency_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate consistency rating based on performance variance"""
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    consistency_stats = []
    
    for player in players:
        player_matches = df[
            (df["Player1"] == player) | (df["Player2"] == player)
        ].copy()
        
        # Get player's scores
        scores = []
        for _, match in player_matches.iterrows():
            if match["Player1"] == player:
                scores.append(match["Score1"])
            else:
                scores.append(match["Score2"])
        
        if len(scores) > 1:
            score_variance = np.var(scores)
            score_std = np.std(scores)
            score_cv = score_std / np.mean(scores) if np.mean(scores) > 0 else 0
            
            # Lower variance = higher consistency
            consistency_rating = 100 - (score_cv * 100)
        else:
            consistency_rating = 50  # Default for single match
        
        consistency_stats.append({
            "Player": player,
            "Matches": len(scores),
            "Avg Score": np.mean(scores) if scores else 0,
            "Score Std Dev": np.std(scores) if len(scores) > 1 else 0,
            "Consistency Rating": consistency_rating
        })
    
    return pd.DataFrame(consistency_stats)


def calculate_momentum_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum and form metrics"""
    df_sorted = df.sort_values(["date", "match_number_total"]).copy()
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    momentum_stats = []
    
    for player in players:
        player_matches = df_sorted[
            (df_sorted["Player1"] == player) | (df_sorted["Player2"] == player)
        ].tail(10)  # Last 10 matches
        
        if len(player_matches) == 0:
            continue
        
        # Recent form
        recent_wins = len(player_matches[player_matches["Winner"] == player])
        recent_form = recent_wins / len(player_matches)
        
        # Momentum (weighted recent performance)
        weights = np.exp(np.linspace(0, 2, len(player_matches)))
        weights = weights / weights.sum()
        
        win_array = (player_matches["Winner"] == player).astype(int).values
        momentum_score = np.dot(weights, win_array) * 100
        
        # Trend direction
        if len(player_matches) >= 3:
            recent_3 = win_array[-3:].mean()
            older = win_array[:-3].mean() if len(win_array) > 3 else 0
            trend = "📈" if recent_3 > older else "📉" if recent_3 < older else "➡️"
        else:
            trend = "➡️"
        
        momentum_stats.append({
            "Player": player,
            "Last 10 Form": f"{recent_wins}/10",
            "Form %": recent_form * 100,
            "Momentum Score": momentum_score,
            "Trend": trend,
            "Hot Streak": "🔥" if recent_wins >= 7 else "❄️" if recent_wins <= 3 else "😐"
        })
    
    return pd.DataFrame(momentum_stats)


def calculate_fatigue_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance decline over multiple daily matches"""
    fatigue_data = []
    
    # Group by date and analyze performance by match number
    for date, day_matches in df.groupby("date"):
        day_sorted = day_matches.sort_values("match_number_day")
        
        for player in set(day_sorted["Player1"]) | set(day_sorted["Player2"]):
            player_day_matches = day_sorted[
                (day_sorted["Player1"] == player) | (day_sorted["Player2"] == player)
            ]
            
            if len(player_day_matches) < 2:
                continue
            
            first_half = player_day_matches.head(len(player_day_matches) // 2)
            second_half = player_day_matches.tail(len(player_day_matches) // 2)
            
            first_wins = len(first_half[first_half["Winner"] == player])
            second_wins = len(second_half[second_half["Winner"] == player])
            
            first_rate = first_wins / len(first_half) if len(first_half) > 0 else 0
            second_rate = second_wins / len(second_half) if len(second_half) > 0 else 0
            
            fatigue_impact = first_rate - second_rate
            
            fatigue_data.append({
                "Player": player,
                "Date": date,
                "Matches Played": len(player_day_matches),
                "Early Win Rate": first_rate,
                "Late Win Rate": second_rate,
                "Fatigue Impact": fatigue_impact,
                "Endurance Score": 100 - (fatigue_impact * 100) if fatigue_impact > 0 else 100
            })
    
    return pd.DataFrame(fatigue_data)


def calculate_nemesis_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Identify nemesis and favorite opponents for each player"""
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    nemesis_data = {}
    
    for player in players:
        opponent_stats = []
        
        for opponent in players:
            if opponent == player:
                continue
            
            h2h_matches = df[
                ((df["Player1"] == player) & (df["Player2"] == opponent)) |
                ((df["Player1"] == opponent) & (df["Player2"] == player))
            ]
            
            if len(h2h_matches) < 3:  # Minimum matches for meaningful analysis
                continue
            
            wins = len(h2h_matches[h2h_matches["Winner"] == player])
            losses = len(h2h_matches[h2h_matches["Loser"] == player])
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            
            # Average margins
            win_margins = h2h_matches[h2h_matches["Winner"] == player]["PointDiff"].mean() if wins > 0 else 0
            loss_margins = abs(h2h_matches[h2h_matches["Loser"] == player]["LoserPointDiff"].mean()) if losses > 0 else 0
            
            # Psychological factor - recent trend
            recent_h2h = h2h_matches.tail(3)
            recent_wins = len(recent_h2h[recent_h2h["Winner"] == player])
            psychological_edge = recent_wins / 3
            
            opponent_stats.append({
                "Opponent": opponent,
                "Matches": len(h2h_matches),
                "Wins": wins,
                "Losses": losses,
                "Win Rate": win_rate,
                "Avg Win Margin": win_margins,
                "Avg Loss Margin": loss_margins,
                "Psychological Edge": psychological_edge,
                "Difficulty Score": (1 - win_rate) * 100 + loss_margins * 2
            })
        
        if opponent_stats:
            nemesis_data[player] = pd.DataFrame(opponent_stats).sort_values("Difficulty Score", ascending=False)
    
    return nemesis_data


def calculate_improvement_velocity(df: pd.DataFrame, rating_history: pd.DataFrame) -> pd.DataFrame:
    """Calculate rate of improvement over time"""
    improvement_stats = []
    
    for player in rating_history["Player"].unique():
        player_ratings = rating_history[rating_history["Player"] == player].sort_values("date")
        
        if len(player_ratings) < 5:
            continue
        
        # Calculate improvement over different time windows
        recent_rating = player_ratings.tail(1).iloc[0]
        month_ago = player_ratings[player_ratings["date"] >= recent_rating["date"] - pd.Timedelta(days=30)]
        three_months_ago = player_ratings[player_ratings["date"] >= recent_rating["date"] - pd.Timedelta(days=90)]
        
        if len(month_ago) > 1:
            month_improvement = recent_rating["Elo Rating"] - month_ago.iloc[0]["Elo Rating"]
        else:
            month_improvement = 0
        
        if len(three_months_ago) > 1:
            quarter_improvement = recent_rating["Elo Rating"] - three_months_ago.iloc[0]["Elo Rating"]
        else:
            quarter_improvement = 0
        
        # Calculate trend
        if len(player_ratings) >= 10:
            x = np.arange(len(player_ratings))
            y = player_ratings["Elo Rating"].values
            slope, _, r_value, _, _ = stats.linregress(x, y)
            improvement_trend = slope
            trend_strength = abs(r_value)
        else:
            improvement_trend = 0
            trend_strength = 0
        
        improvement_stats.append({
            "Player": player,
            "Current Rating": recent_rating["Elo Rating"],
            "1 Month Change": month_improvement,
            "3 Month Change": quarter_improvement,
            "Improvement Velocity": improvement_trend,
            "Trend Strength": trend_strength,
            "Trajectory": "🚀" if improvement_trend > 5 else "📈" if improvement_trend > 0 else "📉" if improvement_trend < -5 else "➡️"
        })
    
    return pd.DataFrame(improvement_stats)


def calculate_pressure_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance under pressure situations"""
    pressure_stats = []
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    
    for player in players:
        # Define pressure situations
        # 1. Match point situations (score >= 10)
        # 2. Revenge matches (match after losing to same opponent)
        # 3. High-stakes matches (both players on winning streaks)
        
        player_matches = df[
            (df["Player1"] == player) | (df["Player2"] == player)
        ].sort_values(["date", "match_number_total"])
        
        pressure_wins = 0
        pressure_total = 0
        
        # Match point situations
        close_matches = player_matches[
            ((player_matches["Score1"] >= 10) | (player_matches["Score2"] >= 10))
        ]
        pressure_total += len(close_matches)
        pressure_wins += len(close_matches[close_matches["Winner"] == player])
        
        # Revenge match analysis
        revenge_opportunities = 0
        revenge_successes = 0
        
        for i in range(1, len(player_matches)):
            current = player_matches.iloc[i]
            
            # Find previous match against same opponent
            if current["Player1"] == player:
                opponent = current["Player2"]
            else:
                opponent = current["Player1"]
            
            previous_vs_opponent = player_matches.iloc[:i][
                ((player_matches.iloc[:i]["Player1"] == player) & (player_matches.iloc[:i]["Player2"] == opponent)) |
                ((player_matches.iloc[:i]["Player1"] == opponent) & (player_matches.iloc[:i]["Player2"] == player))
            ]
            
            if not previous_vs_opponent.empty:
                last_match = previous_vs_opponent.iloc[-1]
                if last_match["Loser"] == player:
                    revenge_opportunities += 1
                    if current["Winner"] == player:
                        revenge_successes += 1
        
        pressure_rating = (pressure_wins / pressure_total * 100) if pressure_total > 0 else 50
        revenge_rate = (revenge_successes / revenge_opportunities * 100) if revenge_opportunities > 0 else 50
        
        pressure_stats.append({
            "Player": player,
            "Pressure Matches": pressure_total,
            "Pressure Wins": pressure_wins,
            "Pressure Win Rate": pressure_wins / pressure_total if pressure_total > 0 else 0,
            "Revenge Opportunities": revenge_opportunities,
            "Revenge Successes": revenge_successes,
            "Revenge Rate": revenge_rate / 100,
            "Mental Fortitude": (pressure_rating + revenge_rate) / 2
        })
    
    return pd.DataFrame(pressure_stats)


def calculate_optimal_rest_period(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance based on rest periods between matches"""
    rest_analysis = []
    players = sorted(set(df["Player1"]) | set(df["Player2"]))
    
    for player in players:
        player_matches = df[
            (df["Player1"] == player) | (df["Player2"] == player)
        ].sort_values("date")
        
        if len(player_matches) < 2:
            continue
        
        performance_by_rest = defaultdict(lambda: {"wins": 0, "total": 0})
        
        for i in range(1, len(player_matches)):
            current = player_matches.iloc[i]
            previous = player_matches.iloc[i-1]
            
            rest_days = (current["date"] - previous["date"]).days
            
            # Categorize rest periods
            if rest_days == 0:
                category = "Same Day"
            elif rest_days <= 3:
                category = "1-3 Days"
            elif rest_days <= 7:
                category = "4-7 Days"
            elif rest_days <= 14:
                category = "1-2 Weeks"
            else:
                category = "2+ Weeks"
            
            performance_by_rest[category]["total"] += 1
            if current["Winner"] == player:
                performance_by_rest[category]["wins"] += 1
        
        # Find optimal rest period
        best_category = ""
        best_rate = 0
        
        for category, stats in performance_by_rest.items():
            if stats["total"] >= 3:  # Minimum sample size
                win_rate = stats["wins"] / stats["total"]
                if win_rate > best_rate:
                    best_rate = win_rate
                    best_category = category
        
        rest_analysis.append({
            "Player": player,
            "Optimal Rest": best_category,
            "Optimal Win Rate": best_rate,
            "Same Day Rate": performance_by_rest["Same Day"]["wins"] / performance_by_rest["Same Day"]["total"] 
                if performance_by_rest["Same Day"]["total"] > 0 else 0,
            "Total Matches Analyzed": len(player_matches) - 1
        })
    
    return pd.DataFrame(rest_analysis)