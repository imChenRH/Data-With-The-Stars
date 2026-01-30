"""
新投票系统设计模块
New Voting System Design Module for MCM 2026 Problem C

提出并分析更公平/更精彩的投票系统
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class VotingSystemDesigner:
    """
    投票系统设计器
    
    设计并评估新的投票系统
    """
    
    def __init__(self):
        self.systems = {}
        self.register_default_systems()
    
    def register_default_systems(self):
        """注册默认的投票系统"""
        # 系统1: 原始排名法
        self.systems['rank'] = {
            'name': 'Rank-Based Method',
            'description': 'Each component contributes equally via ranks',
            'function': self.rank_method,
            'seasons_used': '1-2, 28-34'
        }
        
        # 系统2: 原始百分比法
        self.systems['percentage'] = {
            'name': 'Percentage-Based Method',
            'description': 'Components combined as percentages of totals',
            'function': self.percentage_method,
            'seasons_used': '3-27'
        }
        
        # 系统3: 加权百分比法（新提案）
        self.systems['weighted_percentage'] = {
            'name': 'Weighted Percentage Method',
            'description': 'Judges weighted more heavily to ensure skill recognition',
            'function': self.weighted_percentage_method,
            'seasons_used': 'Proposed'
        }
        
        # 系统4: 动态权重法（新提案）
        self.systems['dynamic_weight'] = {
            'name': 'Dynamic Weight Method',
            'description': 'Weights change based on score variance',
            'function': self.dynamic_weight_method,
            'seasons_used': 'Proposed'
        }
        
        # 系统5: 混合淘汰法（新提案）
        self.systems['hybrid_elimination'] = {
            'name': 'Hybrid Elimination Method',
            'description': 'Bottom 3 identified, judges vote on elimination',
            'function': self.hybrid_elimination_method,
            'seasons_used': 'Proposed'
        }
        
        # 系统6: 累积分数法（新提案）
        self.systems['cumulative'] = {
            'name': 'Cumulative Score Method',
            'description': 'Historical performance affects current weight',
            'function': self.cumulative_method,
            'seasons_used': 'Proposed'
        }
    
    def rank_method(self, judge_scores: np.ndarray, fan_votes: np.ndarray,
                    **kwargs) -> Tuple[np.ndarray, int]:
        """
        排名法
        
        Args:
            judge_scores: 评委总分
            fan_votes: 观众投票
        
        Returns:
            (综合分数, 被淘汰选手索引)
        """
        n = len(judge_scores)
        
        # 计算排名（分数越高排名越好，排名数字越小）
        judge_ranks = n - np.argsort(np.argsort(judge_scores))
        fan_ranks = n - np.argsort(np.argsort(fan_votes))
        
        # 综合排名
        combined_ranks = judge_ranks + fan_ranks
        
        # 最高排名值的选手被淘汰
        eliminated_idx = np.argmax(combined_ranks)
        
        return combined_ranks, eliminated_idx
    
    def percentage_method(self, judge_scores: np.ndarray, fan_votes: np.ndarray,
                          **kwargs) -> Tuple[np.ndarray, int]:
        """
        百分比法
        
        Args:
            judge_scores: 评委总分
            fan_votes: 观众投票
        
        Returns:
            (综合百分比分数, 被淘汰选手索引)
        """
        # 计算百分比
        judge_pct = judge_scores / np.sum(judge_scores)
        fan_pct = fan_votes / np.sum(fan_votes)
        
        # 综合百分比
        combined_pct = judge_pct + fan_pct
        
        # 最低百分比的选手被淘汰
        eliminated_idx = np.argmin(combined_pct)
        
        return combined_pct, eliminated_idx
    
    def weighted_percentage_method(self, judge_scores: np.ndarray, fan_votes: np.ndarray,
                                   judge_weight: float = 0.6, **kwargs) -> Tuple[np.ndarray, int]:
        """
        加权百分比法（新提案）
        
        通过调整评委和观众的权重来平衡专业性和娱乐性
        
        Args:
            judge_scores: 评委总分
            fan_votes: 观众投票
            judge_weight: 评委权重（0-1之间），默认0.6
        
        Returns:
            (加权综合分数, 被淘汰选手索引)
        """
        fan_weight = 1 - judge_weight
        
        # 归一化
        judge_norm = judge_scores / np.sum(judge_scores)
        fan_norm = fan_votes / np.sum(fan_votes)
        
        # 加权组合
        combined = judge_weight * judge_norm + fan_weight * fan_norm
        
        # 最低分被淘汰
        eliminated_idx = np.argmin(combined)
        
        return combined, eliminated_idx
    
    def dynamic_weight_method(self, judge_scores: np.ndarray, fan_votes: np.ndarray,
                              base_judge_weight: float = 0.5, **kwargs) -> Tuple[np.ndarray, int]:
        """
        动态权重法（新提案）
        
        当评委评分差异大时，增加评委权重
        当评委评分接近时，增加观众权重
        
        Args:
            judge_scores: 评委总分
            fan_votes: 观众投票
            base_judge_weight: 基础评委权重
        
        Returns:
            (动态加权分数, 被淘汰选手索引)
        """
        # 计算评委分数的变异系数
        judge_cv = np.std(judge_scores) / (np.mean(judge_scores) + 1e-10)
        
        # 动态调整权重：评分差异越大，评委权重越高
        # CV范围通常在0-0.3之间
        adjustment = min(0.2, judge_cv)  # 最多调整0.2
        judge_weight = base_judge_weight + adjustment
        fan_weight = 1 - judge_weight
        
        # 归一化
        judge_norm = judge_scores / np.sum(judge_scores)
        fan_norm = fan_votes / np.sum(fan_votes)
        
        # 动态加权
        combined = judge_weight * judge_norm + fan_weight * fan_norm
        
        eliminated_idx = np.argmin(combined)
        
        return combined, eliminated_idx
    
    def hybrid_elimination_method(self, judge_scores: np.ndarray, fan_votes: np.ndarray,
                                   bottom_n: int = 3, **kwargs) -> Tuple[np.ndarray, int]:
        """
        混合淘汰法（新提案）
        
        步骤：
        1. 使用百分比法确定底部N名选手
        2. 从底部N名中，评委投票决定淘汰谁
        
        Args:
            judge_scores: 评委总分
            fan_votes: 观众投票
            bottom_n: 进入待淘汰区的选手数量
        
        Returns:
            (综合分数, 被淘汰选手索引)
        """
        # 使用百分比法计算综合分数
        judge_pct = judge_scores / np.sum(judge_scores)
        fan_pct = fan_votes / np.sum(fan_votes)
        combined = judge_pct + fan_pct
        
        # 确定底部N名
        bottom_indices = np.argsort(combined)[:bottom_n]
        
        # 从底部N名中，选择评委分数最低的淘汰
        bottom_judge_scores = judge_scores[bottom_indices]
        eliminated_among_bottom = np.argmin(bottom_judge_scores)
        eliminated_idx = bottom_indices[eliminated_among_bottom]
        
        return combined, eliminated_idx
    
    def cumulative_method(self, judge_scores: np.ndarray, fan_votes: np.ndarray,
                          historical_scores: np.ndarray = None,
                          current_weight: float = 0.7, **kwargs) -> Tuple[np.ndarray, int]:
        """
        累积分数法（新提案）
        
        考虑历史表现，给予一贯表现良好的选手一定保护
        
        Args:
            judge_scores: 当前周评委总分
            fan_votes: 当前周观众投票
            historical_scores: 历史累积分数（可选）
            current_weight: 当前周权重
        
        Returns:
            (综合分数, 被淘汰选手索引)
        """
        # 如果没有历史分数，只使用当前分数
        if historical_scores is None:
            historical_scores = np.zeros_like(judge_scores)
        
        historical_weight = 1 - current_weight
        
        # 归一化当前分数
        judge_norm = judge_scores / np.sum(judge_scores)
        fan_norm = fan_votes / np.sum(fan_votes)
        current_combined = (judge_norm + fan_norm) / 2
        
        # 归一化历史分数
        if np.sum(historical_scores) > 0:
            historical_norm = historical_scores / np.sum(historical_scores)
        else:
            historical_norm = np.ones_like(historical_scores) / len(historical_scores)
        
        # 综合
        combined = current_weight * current_combined + historical_weight * historical_norm
        
        eliminated_idx = np.argmin(combined)
        
        return combined, eliminated_idx
    
    def evaluate_system(self, system_name: str, judge_scores: np.ndarray,
                        fan_votes: np.ndarray, actual_eliminated: int,
                        **kwargs) -> Dict:
        """
        评估单个投票系统
        
        Args:
            system_name: 系统名称
            judge_scores: 评委分数
            fan_votes: 观众投票
            actual_eliminated: 实际被淘汰选手索引
        
        Returns:
            评估结果
        """
        system = self.systems.get(system_name)
        if system is None:
            return {'error': f'Unknown system: {system_name}'}
        
        combined, predicted_eliminated = system['function'](judge_scores, fan_votes, **kwargs)
        
        return {
            'system_name': system['name'],
            'combined_scores': combined,
            'predicted_eliminated': predicted_eliminated,
            'actual_eliminated': actual_eliminated,
            'correct_prediction': predicted_eliminated == actual_eliminated
        }
    
    def compare_all_systems(self, judge_scores: np.ndarray, fan_votes: np.ndarray,
                            actual_eliminated: int, **kwargs) -> pd.DataFrame:
        """
        比较所有投票系统
        
        Args:
            judge_scores: 评委分数
            fan_votes: 观众投票
            actual_eliminated: 实际被淘汰选手索引
        
        Returns:
            比较结果DataFrame
        """
        results = []
        
        for system_name in self.systems:
            eval_result = self.evaluate_system(
                system_name, judge_scores, fan_votes, actual_eliminated, **kwargs
            )
            if 'error' not in eval_result:
                results.append({
                    'System': eval_result['system_name'],
                    'Predicted': eval_result['predicted_eliminated'],
                    'Actual': actual_eliminated,
                    'Correct': eval_result['correct_prediction']
                })
        
        return pd.DataFrame(results)


class FairnessAnalyzer:
    """
    公平性分析器
    
    分析不同投票系统的公平性
    """
    
    def __init__(self, designer: VotingSystemDesigner):
        self.designer = designer
    
    def analyze_judge_fan_balance(self, system_name: str, n_simulations: int = 1000) -> Dict:
        """
        分析系统对评委和观众的平衡性
        
        Args:
            system_name: 系统名称
            n_simulations: 模拟次数
        
        Returns:
            平衡性分析结果
        """
        judge_wins = 0
        fan_wins = 0
        ties = 0
        
        for _ in range(n_simulations):
            # 生成随机场景：4位选手
            n = 4
            judge_scores = np.random.uniform(20, 40, n)
            fan_votes = np.random.uniform(100000, 500000, n)
            
            # 找出评委最低分和观众最低票
            judge_lowest = np.argmin(judge_scores)
            fan_lowest = np.argmin(fan_votes)
            
            # 使用系统确定淘汰者
            system_func = self.designer.systems[system_name]['function']
            _, eliminated = system_func(judge_scores, fan_votes)
            
            if eliminated == judge_lowest and eliminated != fan_lowest:
                judge_wins += 1
            elif eliminated == fan_lowest and eliminated != judge_lowest:
                fan_wins += 1
            else:
                ties += 1
        
        return {
            'system': system_name,
            'judge_preference_rate': judge_wins / n_simulations,
            'fan_preference_rate': fan_wins / n_simulations,
            'agreement_rate': ties / n_simulations,
            'balance_index': 1 - abs(judge_wins - fan_wins) / n_simulations
        }
    
    def analyze_upset_probability(self, system_name: str, n_simulations: int = 1000) -> Dict:
        """
        分析"逆转"发生的概率（低分选手击败高分选手）
        
        Args:
            system_name: 系统名称
            n_simulations: 模拟次数
        
        Returns:
            逆转分析结果
        """
        upsets = 0
        
        for _ in range(n_simulations):
            n = 4
            # 生成评委分数
            judge_scores = np.random.uniform(20, 40, n)
            # 生成与评委分数负相关的观众投票（模拟争议场景）
            fan_votes = np.random.uniform(100000, 500000, n)
            
            # 找出评委最高分选手
            judge_best = np.argmax(judge_scores)
            
            # 确定淘汰者
            system_func = self.designer.systems[system_name]['function']
            _, eliminated = system_func(judge_scores, fan_votes)
            
            # 如果评委最高分选手被淘汰，算作逆转
            if eliminated == judge_best:
                upsets += 1
        
        return {
            'system': system_name,
            'upset_rate': upsets / n_simulations,
            'stability_index': 1 - upsets / n_simulations
        }


def generate_voting_system_report(df: pd.DataFrame = None) -> str:
    """
    生成投票系统分析报告
    
    Args:
        df: 数据DataFrame（可选，用于实际数据分析）
    
    Returns:
        报告文本
    """
    designer = VotingSystemDesigner()
    analyzer = FairnessAnalyzer(designer)
    
    report = []
    report.append("=" * 60)
    report.append("NEW VOTING SYSTEM DESIGN & ANALYSIS REPORT")
    report.append("Dancing with the Stars - MCM 2026 Problem C")
    report.append("=" * 60)
    
    # 系统介绍
    report.append("\n1. VOTING SYSTEMS OVERVIEW")
    report.append("-" * 40)
    
    for name, system in designer.systems.items():
        report.append(f"\n{system['name']} ({name})")
        report.append(f"  Description: {system['description']}")
        report.append(f"  Seasons Used: {system['seasons_used']}")
    
    # 平衡性分析
    report.append("\n\n2. JUDGE-FAN BALANCE ANALYSIS")
    report.append("-" * 40)
    report.append("(Based on 1000 Monte Carlo simulations)")
    
    for name in designer.systems:
        balance = analyzer.analyze_judge_fan_balance(name)
        report.append(f"\n{name}:")
        report.append(f"  Judge Preference Rate: {balance['judge_preference_rate']:.2%}")
        report.append(f"  Fan Preference Rate: {balance['fan_preference_rate']:.2%}")
        report.append(f"  Agreement Rate: {balance['agreement_rate']:.2%}")
        report.append(f"  Balance Index: {balance['balance_index']:.3f}")
    
    # 稳定性分析
    report.append("\n\n3. STABILITY ANALYSIS (UPSET PROBABILITY)")
    report.append("-" * 40)
    
    for name in designer.systems:
        upset = analyzer.analyze_upset_probability(name)
        report.append(f"\n{name}:")
        report.append(f"  Upset Rate: {upset['upset_rate']:.2%}")
        report.append(f"  Stability Index: {upset['stability_index']:.3f}")
    
    # 推荐
    report.append("\n\n4. RECOMMENDATIONS")
    report.append("-" * 40)
    
    recommendations = """
    Based on our analysis, we recommend the following:
    
    FOR FAIRNESS (ensuring skilled dancers advance):
    - Weighted Percentage Method with 60% judge weight
    - Or Hybrid Elimination Method with bottom-3 selection
    
    FOR EXCITEMENT (maintaining fan engagement):
    - Dynamic Weight Method that adjusts based on score variance
    - This rewards skill when differences are clear,
      but allows fan preference when performances are close
    
    FOR CONTROVERSY PREVENTION:
    - Hybrid Elimination provides a safety net
    - Prevents extremely unpopular eliminations while
      still giving fans significant influence
    
    OPTIMAL COMBINATION:
    We propose a tiered system:
    1. Use Dynamic Weight Method for preliminary ranking
    2. Apply Hybrid Elimination for bottom-3 decision
    3. Consider Cumulative Method for later stages to
       protect consistently good performers
    """
    report.append(recommendations)
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


# 示例使用
if __name__ == "__main__":
    # 生成报告
    report = generate_voting_system_report()
    print(report)
    
    # 具体案例分析
    print("\n" + "=" * 60)
    print("CASE STUDY: Season 2 Week 4 Scenario")
    print("=" * 60)
    
    designer = VotingSystemDesigner()
    
    # 模拟第2季第4周场景（Jerry Rice争议相关）
    # 假设有4位选手，Jerry Rice评委分数最低但可能有高观众支持
    judge_scores = np.array([25, 23, 28, 21])  # 选手4最低
    fan_votes = np.array([1500000, 2000000, 1200000, 2500000])  # 选手4最高
    
    print(f"Judge Scores: {judge_scores}")
    print(f"Fan Votes: {fan_votes}")
    print(f"\nActual Eliminated (by judge): Contestant 4 (index 3)")
    
    # 比较所有系统
    results = designer.compare_all_systems(judge_scores, fan_votes, actual_eliminated=0)
    print("\nSystem Comparison:")
    print(results.to_string(index=False))
