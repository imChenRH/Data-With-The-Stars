"""
DWTS Data Analysis Package
Dancing with the Stars MCM 2026 Problem C Analysis
"""

from .data_loader import (
    load_dwts_data,
    clean_data,
    extract_elimination_week,
    get_weekly_standings,
    get_season_info,
    get_judge_consistency
)

__all__ = [
    'load_dwts_data',
    'clean_data', 
    'extract_elimination_week',
    'get_weekly_standings',
    'get_season_info',
    'get_judge_consistency'
]
