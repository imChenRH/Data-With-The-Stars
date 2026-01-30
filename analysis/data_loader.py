"""
数据加载与预处理模块
Data Loading and Preprocessing Module for MCM 2026 Problem C
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_dwts_data(filepath: str = None) -> pd.DataFrame:
    """
    加载DWTS数据文件
    
    Args:
        filepath: CSV文件路径，默认为当前目录下的数据文件
    
    Returns:
        处理后的DataFrame
    """
    if filepath is None:
        base_path = Path(__file__).parent.parent
        filepath = base_path / "2026_MCM_Problem_C_Data.csv"
    
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗
    - 处理N/A值
    - 转换数据类型
    - 提取关键特征
    
    Args:
        df: 原始DataFrame
    
    Returns:
        清洗后的DataFrame
    """
    df = df.copy()
    
    # 提取周分数列名
    score_cols = [col for col in df.columns if 'week' in col.lower() and 'score' in col.lower()]
    
    # 将N/A替换为NaN，并转换为数值类型
    for col in score_cols:
        df[col] = pd.to_numeric(df[col].replace('N/A', np.nan), errors='coerce')
    
    # 确保season和placement为整数
    df['season'] = df['season'].astype(int)
    df['placement'] = pd.to_numeric(df['placement'], errors='coerce')
    
    # 提取淘汰周数
    df['elimination_week'] = df['results'].apply(extract_elimination_week)
    
    # 计算每周总分
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        existing_cols = [col for col in judge_cols if col in df.columns]
        df[f'week{week}_total_score'] = df[existing_cols].sum(axis=1, skipna=True)
        df[f'week{week}_avg_score'] = df[existing_cols].mean(axis=1, skipna=True)
        
        # 有效评委数量
        df[f'week{week}_num_judges'] = df[existing_cols].notna().sum(axis=1) - (df[existing_cols] == 0).sum(axis=1)
    
    return df


def extract_elimination_week(result_str: str) -> int:
    """
    从结果字符串提取淘汰周数
    
    Args:
        result_str: 结果字符串，如 "Eliminated Week 3", "1st Place"
    
    Returns:
        淘汰周数，如果是决赛选手则返回较大值
    """
    if pd.isna(result_str):
        return np.nan
    
    result_str = str(result_str)
    
    if 'Eliminated' in result_str or 'eliminated' in result_str:
        try:
            week = int(result_str.split('Week')[-1].strip())
            return week
        except:
            return np.nan
    elif 'Withdrew' in result_str or 'withdrew' in result_str:
        return -1  # 标记为退出
    elif 'Place' in result_str or 'place' in result_str:
        # 进入决赛的选手
        return 99  # 标记为决赛选手
    else:
        return np.nan


def get_weekly_standings(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """
    获取特定季节特定周的选手排名情况
    
    Args:
        df: 数据DataFrame
        season: 季节编号
        week: 周数
    
    Returns:
        该周的选手排名DataFrame
    """
    season_df = df[df['season'] == season].copy()
    
    # 只保留在该周还未被淘汰的选手
    # 选手在本周或之后被淘汰
    active_contestants = season_df[
        (season_df['elimination_week'] >= week) | 
        (season_df['elimination_week'] == 99) |  # 决赛选手
        (season_df['elimination_week'] == -1)   # 退出选手(根据退出时间判断)
    ]
    
    # 进一步过滤：只保留该周有分数的选手
    score_col = f'week{week}_total_score'
    if score_col in active_contestants.columns:
        active_contestants = active_contestants[active_contestants[score_col] > 0]
    
    return active_contestants


def get_season_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取每个季节的基本信息
    
    Args:
        df: 数据DataFrame
    
    Returns:
        季节信息DataFrame
    """
    season_info = df.groupby('season').agg({
        'celebrity_name': 'count',
        'placement': 'max',
        'elimination_week': 'max'
    }).reset_index()
    
    season_info.columns = ['season', 'num_contestants', 'worst_placement', 'max_weeks']
    
    # 计算实际周数
    for season in season_info['season']:
        season_data = df[df['season'] == season]
        max_week = 1
        for week in range(1, 12):
            score_col = f'week{week}_total_score'
            if score_col in df.columns:
                if (season_data[score_col] > 0).any():
                    max_week = week
        season_info.loc[season_info['season'] == season, 'actual_weeks'] = max_week
    
    return season_info


def get_judge_consistency(df: pd.DataFrame, season: int = None) -> pd.DataFrame:
    """
    分析评委评分一致性
    
    Args:
        df: 数据DataFrame
        season: 可选，特定季节
    
    Returns:
        评委一致性分析DataFrame
    """
    if season is not None:
        df = df[df['season'] == season]
    
    consistency_data = []
    
    for week in range(1, 12):
        judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
        existing_cols = [col for col in judge_cols if col in df.columns]
        
        if len(existing_cols) < 2:
            continue
        
        # 只分析有有效分数的行
        valid_data = df[existing_cols].dropna(how='all')
        valid_data = valid_data[(valid_data > 0).any(axis=1)]
        
        if len(valid_data) > 0:
            # 计算评委间的标准差
            std_between_judges = valid_data.std(axis=1).mean()
            # 计算评委分数的相关性
            if len(existing_cols) >= 2 and len(valid_data) > 1:
                corr_matrix = valid_data.corr()
                avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
            else:
                avg_corr = np.nan
            
            consistency_data.append({
                'week': week,
                'avg_std_between_judges': std_between_judges,
                'avg_correlation': avg_corr,
                'num_contestants': len(valid_data)
            })
    
    return pd.DataFrame(consistency_data)


if __name__ == "__main__":
    # 测试数据加载
    df = load_dwts_data()
    print(f"原始数据形状: {df.shape}")
    
    df_clean = clean_data(df)
    print(f"清洗后数据形状: {df_clean.shape}")
    
    # 显示季节信息
    season_info = get_season_info(df_clean)
    print("\n季节信息:")
    print(season_info.head(10))
    
    # 测试特定周的排名
    standings = get_weekly_standings(df_clean, season=1, week=4)
    print(f"\n第1季第4周活跃选手: {len(standings)}")
    print(standings[['celebrity_name', 'week4_total_score', 'results']])
