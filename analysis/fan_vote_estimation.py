"""
观众投票估算模型
Fan Vote Estimation Model for MCM 2026 Problem C

核心思路:
1. 根据淘汰结果逆向推断观众投票
2. 使用约束优化方法估算满足淘汰条件的投票分布
3. 提供置信区间估计
"""

import pandas as pd
import numpy as np
from scipy import optimize
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class FanVoteEstimator:
    """
    观众投票估算器
    
    使用逆向工程方法，根据已知的评委分数和淘汰结果，
    估算满足这些约束条件的观众投票分布。
    """
    
    def __init__(self, method: str = 'rank'):
        """
        初始化估算器
        
        Args:
            method: 投票合并方法 - 'rank'(排名法) 或 'percentage'(百分比法)
        """
        self.method = method
        self.estimated_votes = {}
        self.confidence_intervals = {}
        
    def compute_combined_score_rank(self, judge_scores: np.ndarray, 
                                     fan_votes: np.ndarray) -> np.ndarray:
        """
        使用排名法计算综合分数
        
        Args:
            judge_scores: 评委总分数组
            fan_votes: 观众投票数组
        
        Returns:
            综合排名分数（越低越好）
        """
        # 计算评委排名（分数高排名好，排名数字小）
        judge_ranks = len(judge_scores) + 1 - np.argsort(np.argsort(judge_scores)) - 1
        
        # 计算观众投票排名
        fan_ranks = len(fan_votes) + 1 - np.argsort(np.argsort(fan_votes)) - 1
        
        # 综合排名 = 评委排名 + 观众排名
        combined_ranks = judge_ranks + fan_ranks
        
        return combined_ranks
    
    def compute_combined_score_percentage(self, judge_scores: np.ndarray,
                                          fan_votes: np.ndarray) -> np.ndarray:
        """
        使用百分比法计算综合分数
        
        Args:
            judge_scores: 评委总分数组
            fan_votes: 观众投票数组
        
        Returns:
            综合百分比分数（越高越好）
        """
        # 计算评委分数百分比
        total_judge = np.sum(judge_scores)
        judge_pct = judge_scores / total_judge if total_judge > 0 else np.zeros_like(judge_scores)
        
        # 计算观众投票百分比
        total_fan = np.sum(fan_votes)
        fan_pct = fan_votes / total_fan if total_fan > 0 else np.zeros_like(fan_votes)
        
        # 综合百分比
        combined_pct = judge_pct + fan_pct
        
        return combined_pct
    
    def estimate_votes_for_week(self, judge_scores: np.ndarray, 
                                 eliminated_idx: int,
                                 n_simulations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        估算某周的观众投票
        
        Args:
            judge_scores: 该周各选手的评委总分
            eliminated_idx: 被淘汰选手的索引
            n_simulations: 模拟次数
        
        Returns:
            (估算的投票比例, 标准差)
        """
        n_contestants = len(judge_scores)
        valid_votes = []
        
        # 使用蒙特卡洛方法采样满足约束的投票分布
        for _ in range(n_simulations):
            # 生成随机投票比例（狄利克雷分布）
            alpha = np.ones(n_contestants)  # 均匀先验
            votes = np.random.dirichlet(alpha)
            
            # 检查是否满足淘汰约束
            if self._check_elimination_constraint(judge_scores, votes, eliminated_idx):
                valid_votes.append(votes)
        
        if len(valid_votes) == 0:
            # 如果没有找到有效样本，使用启发式方法
            votes = self._heuristic_estimate(judge_scores, eliminated_idx)
            return votes, np.ones(n_contestants) * 0.1  # 高不确定性
        
        valid_votes = np.array(valid_votes)
        mean_votes = np.mean(valid_votes, axis=0)
        std_votes = np.std(valid_votes, axis=0)
        
        return mean_votes, std_votes
    
    def _check_elimination_constraint(self, judge_scores: np.ndarray,
                                       fan_votes: np.ndarray,
                                       eliminated_idx: int) -> bool:
        """
        检查投票是否满足淘汰约束
        
        Args:
            judge_scores: 评委分数
            fan_votes: 观众投票比例
            eliminated_idx: 被淘汰选手索引
        
        Returns:
            是否满足约束（被淘汰选手应该是最低分）
        """
        if self.method == 'rank':
            combined = self.compute_combined_score_rank(judge_scores, fan_votes)
            # 排名法中，最高排名值的选手被淘汰
            return np.argmax(combined) == eliminated_idx
        else:
            combined = self.compute_combined_score_percentage(judge_scores, fan_votes)
            # 百分比法中，最低百分比的选手被淘汰
            return np.argmin(combined) == eliminated_idx
    
    def _heuristic_estimate(self, judge_scores: np.ndarray, 
                            eliminated_idx: int) -> np.ndarray:
        """
        启发式投票估算
        
        当蒙特卡洛方法无法找到有效样本时使用
        
        Args:
            judge_scores: 评委分数
            eliminated_idx: 被淘汰选手索引
        
        Returns:
            估算的投票比例
        """
        n = len(judge_scores)
        
        # 基于评委排名的逆向估算
        # 假设观众投票与评委分数有一定相关性但也有差异
        normalized_scores = (judge_scores - np.min(judge_scores)) / (np.max(judge_scores) - np.min(judge_scores) + 1e-10)
        
        # 被淘汰选手获得最低投票
        votes = normalized_scores.copy()
        votes[eliminated_idx] = 0.01  # 给被淘汰者极低投票
        
        # 归一化
        votes = votes / np.sum(votes)
        
        return votes
    
    def estimate_season_votes(self, df: pd.DataFrame, season: int,
                              verbose: bool = False) -> Dict:
        """
        估算整个季节的观众投票
        
        Args:
            df: 包含清洗后数据的DataFrame
            season: 季节编号
            verbose: 是否打印详细信息
        
        Returns:
            包含每周估算结果的字典
        """
        season_data = df[df['season'] == season].copy()
        results = {}
        
        # 获取该季节的周数
        max_week = 1
        for week in range(1, 12):
            score_col = f'week{week}_total_score'
            if score_col in df.columns:
                if (season_data[score_col] > 0).any():
                    max_week = week
        
        if verbose:
            print(f"\n=== Season {season} Analysis ===")
            print(f"Total weeks: {max_week}")
        
        for week in range(1, max_week + 1):
            # 获取该周的活跃选手
            score_col = f'week{week}_total_score'
            active = season_data[season_data[score_col] > 0].copy()
            
            if len(active) <= 1:
                continue
            
            # 找出被淘汰的选手
            eliminated = active[active['elimination_week'] == week]
            
            if len(eliminated) == 0:
                # 本周没有淘汰（例如双倍淘汰周的前一周）
                if verbose:
                    print(f"Week {week}: No elimination")
                continue
            
            eliminated_name = eliminated['celebrity_name'].values[0]
            eliminated_idx = active['celebrity_name'].tolist().index(eliminated_name)
            
            # 获取评委分数
            judge_scores = active[score_col].values
            
            # 估算投票
            mean_votes, std_votes = self.estimate_votes_for_week(
                judge_scores, eliminated_idx, n_simulations=1000
            )
            
            results[week] = {
                'contestants': active['celebrity_name'].tolist(),
                'judge_scores': judge_scores,
                'estimated_votes': mean_votes,
                'vote_std': std_votes,
                'eliminated': eliminated_name,
                'eliminated_idx': eliminated_idx
            }
            
            if verbose:
                print(f"\nWeek {week}:")
                print(f"  Contestants: {len(active)}")
                print(f"  Eliminated: {eliminated_name}")
                print(f"  Judge scores: {judge_scores}")
                print(f"  Estimated vote shares: {np.round(mean_votes * 100, 1)}%")
                print(f"  Vote uncertainty: {np.round(std_votes * 100, 1)}%")
        
        return results


class VotingMethodComparator:
    """
    投票方法比较器
    
    比较排名法和百分比法的效果差异
    """
    
    def __init__(self):
        self.rank_estimator = FanVoteEstimator(method='rank')
        self.pct_estimator = FanVoteEstimator(method='percentage')
    
    def compare_methods(self, judge_scores: np.ndarray, 
                        fan_votes: np.ndarray) -> Dict:
        """
        比较两种方法在给定分数和投票下的结果
        
        Args:
            judge_scores: 评委分数
            fan_votes: 观众投票
        
        Returns:
            比较结果字典
        """
        rank_combined = self.rank_estimator.compute_combined_score_rank(judge_scores, fan_votes)
        pct_combined = self.pct_estimator.compute_combined_score_percentage(judge_scores, fan_votes)
        
        # 排名法：最高值被淘汰
        rank_eliminated = np.argmax(rank_combined)
        
        # 百分比法：最低值被淘汰
        pct_eliminated = np.argmin(pct_combined)
        
        return {
            'rank_scores': rank_combined,
            'pct_scores': pct_combined,
            'rank_eliminated_idx': rank_eliminated,
            'pct_eliminated_idx': pct_eliminated,
            'methods_agree': rank_eliminated == pct_eliminated
        }
    
    def analyze_controversies(self, df: pd.DataFrame, 
                               controversy_cases: List[Dict]) -> pd.DataFrame:
        """
        分析争议案例
        
        Args:
            df: 数据DataFrame
            controversy_cases: 争议案例列表
        
        Returns:
            分析结果DataFrame
        """
        results = []
        
        for case in controversy_cases:
            season = case['season']
            celebrity = case['celebrity']
            
            # 获取该选手数据
            celeb_data = df[(df['season'] == season) & 
                           (df['celebrity_name'].str.contains(celebrity, case=False))]
            
            if len(celeb_data) == 0:
                continue
            
            celeb_row = celeb_data.iloc[0]
            
            # 计算该选手在各周的评委排名
            season_data = df[df['season'] == season]
            
            judge_rank_history = []
            for week in range(1, 12):
                score_col = f'week{week}_total_score'
                if score_col not in df.columns:
                    continue
                
                active = season_data[season_data[score_col] > 0]
                if len(active) == 0 or celeb_row[score_col] == 0:
                    continue
                
                scores = active[score_col].values
                celeb_score = celeb_row[score_col]
                
                # 计算排名（1为最高）
                rank = sum(scores > celeb_score) + 1
                judge_rank_history.append({
                    'week': week,
                    'score': celeb_score,
                    'rank': rank,
                    'total_contestants': len(active)
                })
            
            results.append({
                'season': season,
                'celebrity': celeb_row['celebrity_name'],
                'final_placement': celeb_row['placement'],
                'judge_rank_history': judge_rank_history,
                'num_last_place_weeks': sum(1 for w in judge_rank_history 
                                            if w['rank'] == w['total_contestants'])
            })
        
        return pd.DataFrame(results)


def calculate_consistency_metrics(estimated_votes: Dict, 
                                   actual_eliminations: Dict) -> Dict:
    """
    计算估算一致性指标
    
    Args:
        estimated_votes: 估算的投票结果
        actual_eliminations: 实际淘汰结果
    
    Returns:
        一致性指标字典
    """
    total_weeks = 0
    correct_predictions = 0
    
    for week, data in estimated_votes.items():
        total_weeks += 1
        
        # 找出估算中投票最低的选手
        estimated_eliminated_idx = np.argmin(data['estimated_votes'])
        actual_eliminated_idx = data['eliminated_idx']
        
        if estimated_eliminated_idx == actual_eliminated_idx:
            correct_predictions += 1
    
    return {
        'total_weeks': total_weeks,
        'correct_predictions': correct_predictions,
        'accuracy': correct_predictions / total_weeks if total_weeks > 0 else 0
    }


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data_loader import load_dwts_data, clean_data
    
    # 加载数据
    df = load_dwts_data()
    df = clean_data(df)
    
    # 测试投票估算
    estimator = FanVoteEstimator(method='rank')
    
    # 分析第1季
    results = estimator.estimate_season_votes(df, season=1, verbose=True)
    
    # 分析第2季（Jerry Rice争议季）
    print("\n" + "="*50)
    results = estimator.estimate_season_votes(df, season=2, verbose=True)
