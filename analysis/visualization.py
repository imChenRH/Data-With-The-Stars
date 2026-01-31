"""
数据可视化模块
Data Visualization Module for MCM 2026 Problem C

生成高质量的图表用于论文写作
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和全局样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 定义配色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#555555'
}

# 定义输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "figures"


def ensure_output_dir():
    """确保输出目录存在"""
    OUTPUT_DIR.mkdir(exist_ok=True)


def plot_season_overview(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    绘制季节概览图
    展示各季节的选手数量和周数
    
    Args:
        df: 数据DataFrame
        save: 是否保存图片
    
    Returns:
        matplotlib Figure对象
    """
    ensure_output_dir()
    
    # 统计每季信息
    season_stats = df.groupby('season').agg({
        'celebrity_name': 'count',
        'placement': 'max'
    }).reset_index()
    season_stats.columns = ['Season', 'Contestants', 'Max Placement']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 选手数量随季节变化
    ax1 = axes[0]
    bars = ax1.bar(season_stats['Season'], season_stats['Contestants'], 
                   color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Number of Contestants', fontsize=12)
    ax1.set_title('Number of Contestants per Season', fontsize=14, fontweight='bold')
    ax1.axhline(y=season_stats['Contestants'].mean(), color=COLORS['accent'], 
                linestyle='--', label=f"Mean: {season_stats['Contestants'].mean():.1f}")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 最高名次分布（反映周数）
    ax2 = axes[1]
    ax2.plot(season_stats['Season'], season_stats['Max Placement'], 
             marker='o', color=COLORS['secondary'], linewidth=2, markersize=6)
    ax2.fill_between(season_stats['Season'], season_stats['Max Placement'], 
                     alpha=0.3, color=COLORS['secondary'])
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Worst Placement (Proxy for Duration)', fontsize=12)
    ax2.set_title('Season Duration (by Worst Placement)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'season_overview.png', bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
    
    return fig


def plot_judge_scores_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    绘制评委分数分布图
    
    Args:
        df: 数据DataFrame
        save: 是否保存图片
    
    Returns:
        matplotlib Figure对象
    """
    ensure_output_dir()
    
    # 收集所有评委分数
    all_scores = []
    for week in range(1, 12):
        for judge in range(1, 5):
            col = f'week{week}_judge{judge}_score'
            if col in df.columns:
                scores = df[col].dropna()
                scores = scores[(scores > 0) & (scores <= 10)]
                all_scores.extend(scores.tolist())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 分数直方图
    ax1 = axes[0]
    ax1.hist(all_scores, bins=20, color=COLORS['primary'], alpha=0.7, 
             edgecolor='white', density=True)
    ax1.set_xlabel('Judge Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Judge Scores (All Seasons)', fontsize=14, fontweight='bold')
    ax1.axvline(x=np.mean(all_scores), color=COLORS['accent'], linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_scores):.2f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 按周的分数趋势
    ax2 = axes[1]
    week_means = []
    week_stds = []
    weeks = []
    
    for week in range(1, 12):
        week_scores = []
        for judge in range(1, 5):
            col = f'week{week}_judge{judge}_score'
            if col in df.columns:
                scores = df[col].dropna()
                scores = scores[(scores > 0) & (scores <= 10)]
                week_scores.extend(scores.tolist())
        
        if len(week_scores) > 10:
            weeks.append(week)
            week_means.append(np.mean(week_scores))
            week_stds.append(np.std(week_scores))
    
    ax2.errorbar(weeks, week_means, yerr=week_stds, marker='o', 
                 color=COLORS['secondary'], linewidth=2, markersize=8,
                 capsize=5, capthick=2, ecolor=COLORS['neutral'])
    ax2.set_xlabel('Week', fontsize=12)
    ax2.set_ylabel('Mean Judge Score', fontsize=12)
    ax2.set_title('Judge Score Trends by Week', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'judge_scores_distribution.png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    return fig


def plot_industry_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    绘制行业分析图
    分析不同行业选手的表现
    
    Args:
        df: 数据DataFrame
        save: 是否保存图片
    
    Returns:
        matplotlib Figure对象
    """
    ensure_output_dir()
    
    # 统计行业分布
    industry_stats = df.groupby('celebrity_industry').agg({
        'celebrity_name': 'count',
        'placement': ['mean', 'min'],
        'week1_judge1_score': 'mean'
    }).reset_index()
    industry_stats.columns = ['Industry', 'Count', 'Avg Placement', 'Best Placement', 'Avg First Score']
    industry_stats = industry_stats.sort_values('Count', ascending=False)
    
    # 只保留出现次数超过5次的行业
    industry_stats = industry_stats[industry_stats['Count'] >= 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 行业分布饼图
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(industry_stats)))
    wedges, texts, autotexts = ax1.pie(industry_stats['Count'], 
                                        labels=industry_stats['Industry'],
                                        autopct='%1.1f%%', colors=colors,
                                        pctdistance=0.85)
    ax1.set_title('Celebrity Industry Distribution', fontsize=14, fontweight='bold')
    
    # 图2: 行业数量柱状图
    ax2 = axes[0, 1]
    bars = ax2.barh(industry_stats['Industry'], industry_stats['Count'], 
                    color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax2.set_xlabel('Number of Contestants', fontsize=12)
    ax2.set_title('Contestants by Industry', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 图3: 平均排名对比
    ax3 = axes[1, 0]
    ax3.barh(industry_stats['Industry'], industry_stats['Avg Placement'], 
             color=COLORS['secondary'], alpha=0.8, edgecolor='white')
    ax3.set_xlabel('Average Placement (Lower is Better)', fontsize=12)
    ax3.set_title('Average Placement by Industry', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_xaxis()  # 低排名在右边（更好）
    
    # 图4: 获胜率分析
    ax4 = axes[1, 1]
    win_rates = []
    for industry in industry_stats['Industry']:
        industry_data = df[df['celebrity_industry'] == industry]
        win_rate = (industry_data['placement'] <= 3).sum() / len(industry_data) * 100
        win_rates.append(win_rate)
    
    ax4.barh(industry_stats['Industry'], win_rates, 
             color=COLORS['accent'], alpha=0.8, edgecolor='white')
    ax4.set_xlabel('Top 3 Rate (%)', fontsize=12)
    ax4.set_title('Top 3 Finish Rate by Industry', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'industry_analysis.png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    return fig


def plot_age_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    绘制年龄分析图
    
    Args:
        df: 数据DataFrame
        save: 是否保存图片
    
    Returns:
        matplotlib Figure对象
    """
    ensure_output_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 清理数据
    df_clean = df[df['celebrity_age_during_season'].notna()].copy()
    df_clean = df_clean[df_clean['celebrity_age_during_season'] > 0]
    
    # 图1: 年龄分布
    ax1 = axes[0, 0]
    ax1.hist(df_clean['celebrity_age_during_season'], bins=25, 
             color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax1.axvline(x=df_clean['celebrity_age_during_season'].mean(), 
                color=COLORS['accent'], linestyle='--', linewidth=2,
                label=f"Mean: {df_clean['celebrity_age_during_season'].mean():.1f}")
    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Age Distribution of Contestants', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 年龄 vs 最终排名
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df_clean['celebrity_age_during_season'], 
                          df_clean['placement'], 
                          c=df_clean['season'], cmap='viridis',
                          alpha=0.6, s=50, edgecolors='white')
    ax2.set_xlabel('Age', fontsize=12)
    ax2.set_ylabel('Final Placement', fontsize=12)
    ax2.set_title('Age vs Final Placement', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Season')
    ax2.grid(alpha=0.3)
    
    # 图3: 年龄分组分析
    ax3 = axes[1, 0]
    df_clean['age_group'] = pd.cut(df_clean['celebrity_age_during_season'], 
                                    bins=[0, 25, 35, 45, 55, 100],
                                    labels=['<25', '25-35', '35-45', '45-55', '55+'])
    age_placement = df_clean.groupby('age_group', observed=True)['placement'].mean()
    
    bars = ax3.bar(age_placement.index.astype(str), age_placement.values, 
                   color=COLORS['secondary'], alpha=0.8, edgecolor='white')
    ax3.set_xlabel('Age Group', fontsize=12)
    ax3.set_ylabel('Average Placement', fontsize=12)
    ax3.set_title('Average Placement by Age Group', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 图4: 年龄与第一周分数
    ax4 = axes[1, 1]
    
    # 计算第一周平均分
    first_week_cols = [col for col in df_clean.columns if 'week1_judge' in col and 'score' in col]
    df_clean['week1_avg'] = df_clean[first_week_cols].mean(axis=1)
    df_clean = df_clean[df_clean['week1_avg'] > 0]
    
    ax4.scatter(df_clean['celebrity_age_during_season'], df_clean['week1_avg'],
                alpha=0.5, color=COLORS['accent'], s=50, edgecolors='white')
    
    # 添加趋势线
    z = np.polyfit(df_clean['celebrity_age_during_season'], df_clean['week1_avg'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_clean['celebrity_age_during_season'].min(), 
                         df_clean['celebrity_age_during_season'].max(), 100)
    ax4.plot(x_line, p(x_line), color=COLORS['success'], linewidth=2, 
             linestyle='--', label=f'Trend: slope={z[0]:.3f}')
    
    ax4.set_xlabel('Age', fontsize=12)
    ax4.set_ylabel('Week 1 Average Score', fontsize=12)
    ax4.set_title('Age vs First Week Performance', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'age_analysis.png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    return fig


def plot_pro_partner_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    绘制专业舞伴分析图
    
    Args:
        df: 数据DataFrame
        save: 是否保存图片
    
    Returns:
        matplotlib Figure对象
    """
    ensure_output_dir()
    
    # 统计舞伴数据
    partner_stats = df.groupby('ballroom_partner').agg({
        'celebrity_name': 'count',
        'placement': ['mean', 'min', 'std']
    }).reset_index()
    partner_stats.columns = ['Partner', 'Appearances', 'Avg Placement', 'Best Placement', 'Placement Std']
    partner_stats = partner_stats.sort_values('Appearances', ascending=False)
    
    # 只保留出现次数超过3次的舞伴
    partner_stats = partner_stats[partner_stats['Appearances'] >= 3].head(15)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 图1: 舞伴出场次数
    ax1 = axes[0, 0]
    bars = ax1.barh(partner_stats['Partner'], partner_stats['Appearances'],
                    color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax1.set_xlabel('Number of Seasons', fontsize=12)
    ax1.set_title('Most Frequent Professional Partners', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 图2: 平均排名对比
    ax2 = axes[0, 1]
    sorted_by_placement = partner_stats.sort_values('Avg Placement')
    ax2.barh(sorted_by_placement['Partner'], sorted_by_placement['Avg Placement'],
             color=COLORS['secondary'], alpha=0.8, edgecolor='white')
    ax2.set_xlabel('Average Placement (Lower is Better)', fontsize=12)
    ax2.set_title('Professional Partners by Average Placement', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 图3: 冠军次数
    ax3 = axes[1, 0]
    winner_counts = df[df['placement'] == 1].groupby('ballroom_partner').size()
    winner_counts = winner_counts.sort_values(ascending=False).head(10)
    
    if len(winner_counts) > 0:
        ax3.barh(winner_counts.index, winner_counts.values,
                 color=COLORS['accent'], alpha=0.8, edgecolor='white')
        ax3.set_xlabel('Number of Wins', fontsize=12)
        ax3.set_title('Professional Partners with Most Wins', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
    
    # 图4: 排名稳定性（标准差）
    ax4 = axes[1, 1]
    sorted_by_std = partner_stats.sort_values('Placement Std')
    colors_std = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_by_std)))
    ax4.barh(sorted_by_std['Partner'], sorted_by_std['Placement Std'], color=colors_std)
    ax4.set_xlabel('Placement Standard Deviation', fontsize=12)
    ax4.set_title('Partner Consistency (Lower = More Consistent)', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'pro_partner_analysis.png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    return fig


def plot_controversy_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    绘制争议案例分析图
    
    Args:
        df: 数据DataFrame
        save: 是否保存图片
    
    Returns:
        matplotlib Figure对象
    """
    ensure_output_dir()
    
    # 定义争议案例
    controversy_cases = [
        ('Jerry Rice', 2, 'Runner-up with lowest scores in 5 weeks'),
        ('Billy Ray Cyrus', 4, '5th despite last place in 6 weeks'),
        ('Bristol Palin', 11, '3rd with lowest scores 12 times'),
        ('Bobby Bones', 27, 'Won despite consistently low scores')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, (name, season, description) in enumerate(controversy_cases):
        ax = axes[idx // 2, idx % 2]
        
        # 获取该选手数据
        celeb_data = df[(df['season'] == season) & 
                        (df['celebrity_name'].str.contains(name, case=False, na=False))]
        
        if len(celeb_data) == 0:
            ax.text(0.5, 0.5, f'Data not found for {name} (Season {season})', 
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        celeb_row = celeb_data.iloc[0]
        season_data = df[df['season'] == season]
        
        # 收集该选手每周的分数和排名
        weeks = []
        celeb_scores = []
        celeb_ranks = []
        
        for week in range(1, 12):
            score_col = f'week{week}_total_score'
            if score_col not in df.columns:
                continue
            
            # 获取该周活跃选手
            active = season_data[season_data[score_col] > 0]
            if len(active) == 0 or celeb_row[score_col] == 0:
                continue
            
            weeks.append(week)
            celeb_scores.append(celeb_row[score_col])
            
            # 计算排名
            rank = (active[score_col] > celeb_row[score_col]).sum() + 1
            celeb_ranks.append(rank)
        
        if len(weeks) == 0:
            continue
        
        # 绘制双轴图
        color1 = COLORS['primary']
        color2 = COLORS['secondary']
        
        ax.bar(weeks, celeb_scores, color=color1, alpha=0.7, label='Total Score')
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Total Judge Score', color=color1, fontsize=10)
        ax.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax.twinx()
        ax2.plot(weeks, celeb_ranks, color=color2, marker='s', linewidth=2, 
                 markersize=8, label='Rank')
        ax2.set_ylabel('Weekly Rank (1=Best)', color=color2, fontsize=10)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.invert_yaxis()
        
        ax.set_title(f'{name} - Season {season}\n{description}\nFinal: {celeb_row["placement"]}', 
                     fontsize=11, fontweight='bold')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'controversy_analysis.png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    return fig


def plot_voting_method_comparison(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    绘制投票方法比较图
    
    Args:
        df: 数据DataFrame
        save: 是否保存图片
    
    Returns:
        matplotlib Figure对象
    """
    ensure_output_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 投票方法使用时间线
    ax1 = axes[0, 0]
    seasons = np.arange(1, 35)
    rank_method = [1 if s in [1, 2] or s >= 28 else 0 for s in seasons]
    pct_method = [1 if 3 <= s <= 27 else 0 for s in seasons]
    
    ax1.fill_between(seasons, 0, rank_method, alpha=0.7, label='Rank Method', 
                     color=COLORS['primary'])
    ax1.fill_between(seasons, 0, [-x for x in pct_method], alpha=0.7, 
                     label='Percentage Method', color=COLORS['secondary'])
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Method in Use', fontsize=12)
    ax1.set_title('Voting Method Usage Over Seasons', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(['Percentage', '', 'Rank'])
    
    # 图2: 季节3-27 vs 1-2,28-34 比较
    ax2 = axes[0, 1]
    
    rank_seasons = df[(df['season'].isin([1, 2])) | (df['season'] >= 28)]
    pct_seasons = df[(df['season'] >= 3) & (df['season'] <= 27)]
    
    # 计算平均排名差异
    data_to_plot = [
        rank_seasons['placement'].dropna(),
        pct_seasons['placement'].dropna()
    ]
    
    bp = ax2.boxplot(data_to_plot, labels=['Rank Method\n(S1-2, S28-34)', 
                                            'Percentage Method\n(S3-27)'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['primary'])
    bp['boxes'][1].set_facecolor(COLORS['secondary'])
    
    ax2.set_ylabel('Final Placement', fontsize=12)
    ax2.set_title('Placement Distribution by Voting Method Era', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 图3: 概念性对比 - 评委影响力
    ax3 = axes[1, 0]
    
    # 模拟数据展示两种方法的差异
    judge_scores = np.array([28, 26, 24, 22])  # 4位选手的评委分
    fan_vote_scenarios = [
        [0.35, 0.25, 0.25, 0.15],  # 场景1: 高分选手获得更多票
        [0.15, 0.25, 0.25, 0.35],  # 场景2: 低分选手获得更多票
        [0.25, 0.25, 0.25, 0.25],  # 场景3: 票数均匀分布
    ]
    
    x = np.arange(4)
    width = 0.2
    
    for i, fan_votes in enumerate(fan_vote_scenarios):
        fan_votes = np.array(fan_votes)
        
        # 排名法计算
        judge_ranks = 4 - np.argsort(np.argsort(judge_scores))
        fan_ranks = 4 - np.argsort(np.argsort(fan_votes))
        rank_combined = judge_ranks + fan_ranks
        
        # 百分比法计算
        judge_pct = judge_scores / judge_scores.sum()
        fan_pct = fan_votes
        pct_combined = judge_pct + fan_pct
        
        ax3.bar(x + i*width, rank_combined, width, alpha=0.8, 
                label=f'Scenario {i+1}')
    
    ax3.set_xlabel('Contestant', fontsize=12)
    ax3.set_ylabel('Combined Rank Score (Lower = Better)', fontsize=12)
    ax3.set_title('Rank Method: Combined Scores by Scenario', fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([f'C{i+1}\n(Judge:{s})' for i, s in enumerate(judge_scores)])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 图4: 方法公平性分析文字
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    comparison_text = """
    Voting Method Comparison Summary
    ================================
    
    RANK METHOD (Seasons 1-2, 28-34):
    • Each vote type contributes equally (1 rank point)
    • Protects against extreme outliers
    • May reduce impact of large score differences
    • More resistant to vote manipulation
    
    PERCENTAGE METHOD (Seasons 3-27):
    • Proportional representation
    • Large differences in scores/votes matter more
    • Can amplify fan preference over judges
    • Higher sensitivity to vote distribution
    
    KEY FINDINGS:
    • Percentage method tends to favor fan votes
    • Rank method provides more stability
    • Controversy cases often involve methods
      that amplify fan-judge disagreements
    """
    
    ax4.text(0.1, 0.95, comparison_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'voting_method_comparison.png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    return fig


def plot_heatmap_judge_correlation(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    绘制评委分数相关性热力图
    
    Args:
        df: 数据DataFrame
        save: 是否保存图片
    
    Returns:
        matplotlib Figure对象
    """
    ensure_output_dir()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 收集所有第一周评委分数
    week1_data = df[['week1_judge1_score', 'week1_judge2_score', 'week1_judge3_score', 'week1_judge4_score']].copy()
    week1_data = week1_data.replace('N/A', np.nan)
    for col in week1_data.columns:
        week1_data[col] = pd.to_numeric(week1_data[col], errors='coerce')
    week1_data = week1_data[(week1_data > 0).all(axis=1)]
    
    if len(week1_data) > 10:
        corr_matrix = week1_data.corr()
        
        ax1 = axes[0]
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', vmin=0, vmax=1,
                    ax=ax1, fmt='.2f', square=True,
                    xticklabels=['Judge 1', 'Judge 2', 'Judge 3', 'Judge 4'],
                    yticklabels=['Judge 1', 'Judge 2', 'Judge 3', 'Judge 4'])
        ax1.set_title('Judge Score Correlation (Week 1, All Seasons)', fontsize=14, fontweight='bold')
    
    # 按季节分析评委一致性
    ax2 = axes[1]
    
    season_consistency = []
    for season in df['season'].unique():
        season_data = df[df['season'] == season]
        week1_scores = season_data[['week1_judge1_score', 'week1_judge2_score', 'week1_judge3_score']].copy()
        week1_scores = week1_scores.replace('N/A', np.nan)
        for col in week1_scores.columns:
            week1_scores[col] = pd.to_numeric(week1_scores[col], errors='coerce')
        week1_scores = week1_scores[(week1_scores > 0).all(axis=1)]
        
        if len(week1_scores) > 3:
            std = week1_scores.std(axis=1).mean()
            season_consistency.append({'season': season, 'avg_std': std})
    
    if len(season_consistency) > 0:
        consistency_df = pd.DataFrame(season_consistency)
        ax2.plot(consistency_df['season'], consistency_df['avg_std'], 
                 marker='o', color=COLORS['primary'], linewidth=2)
        ax2.fill_between(consistency_df['season'], consistency_df['avg_std'], 
                         alpha=0.3, color=COLORS['primary'])
        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('Average Std Dev Between Judges', fontsize=12)
        ax2.set_title('Judge Scoring Consistency Over Seasons', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(OUTPUT_DIR / 'judge_correlation_heatmap.png', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    
    return fig


def generate_all_figures(df: pd.DataFrame):
    """
    生成所有分析图表
    
    Args:
        df: 清洗后的数据DataFrame
    """
    print("Generating all figures...")
    
    print("1/7 Season Overview...")
    plot_season_overview(df)
    
    print("2/7 Judge Scores Distribution...")
    plot_judge_scores_distribution(df)
    
    print("3/7 Industry Analysis...")
    plot_industry_analysis(df)
    
    print("4/7 Age Analysis...")
    plot_age_analysis(df)
    
    print("5/7 Professional Partner Analysis...")
    plot_pro_partner_analysis(df)
    
    print("6/7 Controversy Analysis...")
    plot_controversy_analysis(df)
    
    print("7/7 Voting Method Comparison...")
    plot_voting_method_comparison(df)
    
    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dwts_data, clean_data
    
    # 加载和清洗数据
    df = load_dwts_data()
    df = clean_data(df)
    
    # 生成所有图表
    generate_all_figures(df)
    
    print("\nVisualization complete!")
