import streamlit as st

def create_info_box(title: str, content: str, expanded: bool = False):
    """
    Creates an expandable info box with a title and detailed content.
    
    Args:
        title (str): The title/header for the info box
        content (str): The detailed explanation content (supports markdown)
        expanded (bool): Whether the box should be expanded by default
    """
    with st.expander(f"ℹ️ {title}", expanded=expanded):
        st.markdown(content)

def create_section_info(section_name: str, description: str, metrics_explanation: str = None, expanded: bool = False):
    """
    Creates a standardized info box for analysis sections.
    
    Args:
        section_name (str): Name of the analysis section
        description (str): What this section analyzes
        metrics_explanation (str): Optional detailed explanation of metrics/calculations
        expanded (bool): Whether to expand by default
    """
    content = f"**What this section analyzes:**\n\n{description}"
    
    if metrics_explanation:
        content += f"\n\n**Key Metrics & Calculations:**\n\n{metrics_explanation}"
    
    create_info_box(f"About {section_name}", content, expanded)

# Predefined explanations for each major section
SECTION_EXPLANATIONS = {
    "Performance Overview": {
        "description": """This section provides a comprehensive snapshot of key performance indicators across all players. It combines multiple metrics into visual summaries that highlight top performers and overall trends.""",
        "metrics": """
• **Key Metrics Table**: Shows win rate, total wins, points scored, and latest ratings for quick comparison
• **Sparklines**: Mini line charts showing recent performance trends for each player
• **Bubble Chart**: Visualizes the relationship between different performance dimensions (e.g., consistency vs. dominance)
• **Performance Indicators**: Color-coded metrics that instantly show who's performing well
        """
    },
    
    "Ratings": {
        "description": """Advanced rating systems that go beyond simple win-loss records to provide sophisticated skill assessments. Each system uses different mathematical approaches to rank players.""",
        "metrics": """
• **Glicko2**: Considers rating reliability and time decay. More accurate for players with fewer games
• **Elo**: Classic chess rating system adapted for squash. Simple but effective for head-to-head comparisons  
• **TrueSkill**: Microsoft's system designed for multiplayer games. Accounts for skill uncertainty
• **Rating Over Time**: Track how each player's skill level evolves throughout the season
        """
    },
    
    "Wins & Points": {
        "description": """Fundamental performance metrics focusing on match outcomes and point accumulation. These are the core statistics that determine tournament standings and overall success.""",
        "metrics": """
• **Match Day Wins**: Victories counted per playing session/day
• **Total Win Rate**: Percentage of matches won (most important metric for ranking)
• **Total Wins**: Absolute number of matches won (shows activity level)
• **Total Points**: Points scored across all matches (indicates offensive capability)
• **Trends Over Time**: See how performance evolves throughout the season
        """
    },
    
    "Avg. Margin": {
        "description": """Analyzes how decisively players win or lose their matches. Margin of victory/defeat reveals playing style and competitiveness level.""",
        "metrics": """
• **Average Margin of Victory**: How many points players typically win by (indicates dominance)
• **Average Margin of Defeat**: How many points players typically lose by (shows resilience)
• **Distribution Analysis**: Whether players have close matches or blowouts
• **Competitive Balance**: Identifies players who consistently have tight matches vs. those with decisive outcomes
        """
    },
    
    "Win/Loss Streaks": {
        "description": """Tracks momentum and consistency by analyzing sequences of consecutive wins or losses. Streaks reveal psychological strength and form cycles.""",
        "metrics": """
• **Longest Win Streak**: Maximum consecutive victories (shows peak performance periods)
• **Longest Loss Streak**: Maximum consecutive defeats (identifies struggling periods)
• **Current Streak**: Whether player is currently on a winning or losing run
• **Streak Frequency**: How often players go on runs vs. alternating results
• **Streak Timeline**: Visual representation of when streaks occurred throughout the season
        """
    },
    
    "Endurance and Grit": {
        "description": """Measures how players perform under physical and mental pressure, particularly in high-stakes or physically demanding situations.""",
        "metrics": """
• **N-th Match Performance**: How win rate changes as players play multiple matches in one day
• **Nerves of Steel**: Performance in 11:9 or 9:11 matches (very close games)
• **Nerves of Adamantium**: Performance in ≥12:10 or ≥10:12 matches (extremely close games)
• **Fatigue Analysis**: Whether players maintain performance levels throughout long playing sessions
• **Clutch Factor**: Success rate in high-pressure, close-score situations
        """
    },
    
    "Records & Leaderboards": {
        "description": """Comprehensive rankings and achievements across all measurable categories. This section highlights exceptional performances and milestones.""",
        "metrics": """
• **Multiple Leaderboard Categories**: Rankings for wins, points, margins, streaks, etc.
• **Record Holders**: Players who hold various statistical records
• **Achievement Milestones**: Notable accomplishments and benchmarks reached
• **Historical Comparisons**: How current performance compares to past seasons
• **Exceptional Performances**: Standout individual match or streak achievements
        """
    },
    
    "Match Stats": {
        "description": """Detailed analysis of match patterns, timing, and distribution. Reveals when and how often matches are played, plus intensity levels.""",
        "metrics": """
• **Matches Over Time**: Frequency and distribution of matches throughout the season
• **Match Distribution**: How matches are spread across players and time periods
• **Match Intensity**: Analysis of point totals and competitiveness levels
• **Day-of-Week Performance**: Whether players perform better on certain days
• **Monthly/Yearly Trends**: Seasonal patterns and long-term performance cycles
• **Time-Based Analysis**: Performance variations based on when matches are played
        """
    },
    
    "Predictive Analytics": {
        "description": """Forward-looking analysis that uses historical data to forecast future performance and identify potential upsets or trends.""",
        "metrics": """
• **Match Prediction**: Probability calculations for future head-to-head matchups
• **Score Forecasting**: Predicted score ranges based on historical performance
• **Upset Potential**: Identification of matches where lower-ranked players might win
• **Performance Trajectory**: Whether players are improving, declining, or stable
• **Trend Analysis**: Statistical patterns that indicate future performance direction
        """
    },
    
    "Performance Patterns": {
        "description": """Advanced behavioral analysis that identifies playing patterns, psychological tendencies, and situational performance variations.""",
        "metrics": """
• **Clutch Performance**: Success rate in pressure situations and close matches
• **Comeback Analysis**: Ability to recover from deficit positions
• **Fatigue Patterns**: How performance changes with physical tiredness
• **Optimal Rest**: Analysis of how rest periods between matches affect performance
• **Situational Awareness**: Performance in different match contexts and scenarios
        """
    },
    
    "Psychological Insights": {
        "description": """Deep dive into the mental game, examining psychological factors that influence performance and player matchups.""",
        "metrics": """
• **Pressure Performance**: How players handle high-stakes or important matches
• **Nemesis Analysis**: Identification of players who consistently struggle against specific opponents
• **Momentum Tracking**: How winning/losing affects subsequent match performance
• **Mental Resilience**: Recovery patterns after losses or difficult matches
• **Psychological Matchups**: Head-to-head dynamics that go beyond pure skill
        """
    },
    
    "Multi-Player Comparison": {
        "description": """Side-by-side analysis of multiple players across various performance dimensions. Enables comprehensive evaluation of relative strengths and weaknesses.""",
        "metrics": """
• **Radar Charts**: Multi-dimensional skill comparisons across key performance areas
• **Comparison Matrices**: Direct statistical comparisons between selected players
• **Relative Performance**: How players stack up against each other in different categories
• **Strengths/Weaknesses**: Identification of each player's competitive advantages and areas for improvement
        """
    },
    
    "Head-to-Head Deep Dive": {
        "description": """Detailed analysis of specific player matchups, revealing tactical patterns and psychological dynamics between individual opponents.""",
        "metrics": """
• **Historical Record**: Complete head-to-head win-loss record between specific players
• **Match Patterns**: Typical score patterns and competitiveness levels in this matchup
• **Tactical Analysis**: How these specific players match up stylistically
• **Trend Analysis**: Whether the historical advantage is shifting over time
• **Psychological Edge**: Evidence of mental advantages one player may have over another
        """
    }
}