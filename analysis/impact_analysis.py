"""
影响因素分析模块
Impact Factor Analysis Module for MCM 2026 Problem C

分析专业舞伴、名人特征等因素对比赛结果的影响
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class ImpactAnalyzer:
    """
    影响因素分析器
    
    分析各种因素对选手表现的影响
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化分析器
        
        Args:
            df: 清洗后的数据DataFrame
        """
        self.df = df.copy()
        self.prepare_features()
    
    def prepare_features(self):
        """准备分析所需的特征"""
        # 编码分类变量
        self.le_industry = LabelEncoder()
        self.le_partner = LabelEncoder()
        self.le_country = LabelEncoder()
        
        # 处理缺失值
        self.df['celebrity_industry'] = self.df['celebrity_industry'].fillna('Unknown')
        self.df['ballroom_partner'] = self.df['ballroom_partner'].fillna('Unknown')
        self.df['celebrity_homecountry/region'] = self.df['celebrity_homecountry/region'].fillna('Unknown')
        
        self.df['industry_encoded'] = self.le_industry.fit_transform(self.df['celebrity_industry'])
        self.df['partner_encoded'] = self.le_partner.fit_transform(self.df['ballroom_partner'])
        self.df['country_encoded'] = self.le_country.fit_transform(self.df['celebrity_homecountry/region'])
        
        # 计算第一周平均分作为初始表现指标
        first_week_cols = [col for col in self.df.columns if 'week1_judge' in col and 'score' in col]
        self.df['week1_avg_score'] = self.df[first_week_cols].mean(axis=1)
        
        # 是否为美国选手
        self.df['is_us'] = (self.df['celebrity_homecountry/region'] == 'United States').astype(int)
    
    def analyze_partner_impact(self) -> dict:
        """
        分析专业舞伴对选手表现的影响
        
        Returns:
            分析结果字典
        """
        results = {}
        
        # 1. 舞伴统计
        partner_stats = self.df.groupby('ballroom_partner').agg({
            'celebrity_name': 'count',
            'placement': ['mean', 'std', 'min'],
            'week1_avg_score': 'mean'
        }).reset_index()
        partner_stats.columns = ['Partner', 'Appearances', 'Avg_Placement', 
                                  'Placement_Std', 'Best_Placement', 'Avg_Week1_Score']
        partner_stats = partner_stats[partner_stats['Appearances'] >= 3]
        partner_stats = partner_stats.sort_values('Avg_Placement')
        
        results['partner_statistics'] = partner_stats
        
        # 2. ANOVA测试 - 舞伴对排名的影响
        qualified_partners = partner_stats[partner_stats['Appearances'] >= 3]['Partner'].tolist()
        groups = [self.df[self.df['ballroom_partner'] == p]['placement'].dropna() 
                  for p in qualified_partners]
        groups = [g for g in groups if len(g) >= 2]
        
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            results['anova_partner_placement'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # 3. 冠军概率分析
        wins_by_partner = self.df[self.df['placement'] == 1].groupby('ballroom_partner').size()
        total_by_partner = self.df.groupby('ballroom_partner').size()
        win_rate = (wins_by_partner / total_by_partner).dropna().sort_values(ascending=False)
        results['win_rates'] = win_rate.head(10).to_dict()
        
        # 4. 前三名概率分析
        top3_by_partner = self.df[self.df['placement'] <= 3].groupby('ballroom_partner').size()
        top3_rate = (top3_by_partner / total_by_partner).dropna().sort_values(ascending=False)
        results['top3_rates'] = top3_rate.head(10).to_dict()
        
        return results
    
    def analyze_age_impact(self) -> dict:
        """
        分析年龄对选手表现的影响
        
        Returns:
            分析结果字典
        """
        results = {}
        
        # 过滤有效年龄数据
        df_age = self.df[self.df['celebrity_age_during_season'].notna() & 
                        (self.df['celebrity_age_during_season'] > 0)].copy()
        
        # 1. 相关性分析
        corr_placement, p_placement = stats.pearsonr(
            df_age['celebrity_age_during_season'], 
            df_age['placement'].dropna()
        )
        
        # 过滤有效的week1分数
        df_week1 = df_age[df_age['week1_avg_score'] > 0]
        if len(df_week1) > 10:
            corr_score, p_score = stats.pearsonr(
                df_week1['celebrity_age_during_season'],
                df_week1['week1_avg_score']
            )
        else:
            corr_score, p_score = np.nan, np.nan
        
        results['correlations'] = {
            'age_vs_placement': {'correlation': corr_placement, 'p_value': p_placement},
            'age_vs_week1_score': {'correlation': corr_score, 'p_value': p_score}
        }
        
        # 2. 年龄分组分析
        df_age['age_group'] = pd.cut(
            df_age['celebrity_age_during_season'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['Under 25', '25-35', '35-45', '45-55', 'Over 55']
        )
        
        age_group_stats = df_age.groupby('age_group', observed=True).agg({
            'celebrity_name': 'count',
            'placement': ['mean', 'std'],
            'week1_avg_score': 'mean'
        }).reset_index()
        age_group_stats.columns = ['Age_Group', 'Count', 'Avg_Placement', 
                                    'Placement_Std', 'Avg_Week1_Score']
        
        results['age_group_statistics'] = age_group_stats.to_dict('records')
        
        # 3. 最优年龄范围分析
        winner_ages = df_age[df_age['placement'] == 1]['celebrity_age_during_season']
        results['winner_age_stats'] = {
            'mean': winner_ages.mean(),
            'median': winner_ages.median(),
            'std': winner_ages.std(),
            'min': winner_ages.min(),
            'max': winner_ages.max()
        }
        
        return results
    
    def analyze_industry_impact(self) -> dict:
        """
        分析名人行业对选手表现的影响
        
        Returns:
            分析结果字典
        """
        results = {}
        
        # 1. 行业统计
        industry_stats = self.df.groupby('celebrity_industry').agg({
            'celebrity_name': 'count',
            'placement': ['mean', 'std', 'min'],
            'week1_avg_score': 'mean'
        }).reset_index()
        industry_stats.columns = ['Industry', 'Count', 'Avg_Placement', 
                                   'Placement_Std', 'Best_Placement', 'Avg_Week1_Score']
        industry_stats = industry_stats[industry_stats['Count'] >= 5]
        industry_stats = industry_stats.sort_values('Avg_Placement')
        
        results['industry_statistics'] = industry_stats.to_dict('records')
        
        # 2. 卡方检验 - 行业与获胜的关联
        df_qualified = self.df[self.df['celebrity_industry'].isin(
            industry_stats['Industry'].tolist()
        )].copy()
        df_qualified['is_top3'] = (df_qualified['placement'] <= 3).astype(int)
        
        contingency_table = pd.crosstab(df_qualified['celebrity_industry'], 
                                        df_qualified['is_top3'])
        if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            results['chi2_industry_top3'] = {
                'chi2': chi2,
                'p_value': p_value,
                'dof': dof,
                'significant': p_value < 0.05
            }
        
        # 3. 行业获胜率
        wins_by_industry = self.df[self.df['placement'] == 1].groupby('celebrity_industry').size()
        total_by_industry = self.df.groupby('celebrity_industry').size()
        industry_win_rate = (wins_by_industry / total_by_industry).dropna()
        results['industry_win_rates'] = industry_win_rate.sort_values(ascending=False).head(10).to_dict()
        
        return results
    
    def analyze_home_advantage(self) -> dict:
        """
        分析地域因素（是否美国）对选手表现的影响
        
        Returns:
            分析结果字典
        """
        results = {}
        
        # 美国 vs 非美国选手对比
        us_contestants = self.df[self.df['is_us'] == 1]
        non_us_contestants = self.df[self.df['is_us'] == 0]
        
        results['us_stats'] = {
            'count': len(us_contestants),
            'avg_placement': us_contestants['placement'].mean(),
            'avg_week1_score': us_contestants['week1_avg_score'].mean(),
            'win_count': (us_contestants['placement'] == 1).sum()
        }
        
        results['non_us_stats'] = {
            'count': len(non_us_contestants),
            'avg_placement': non_us_contestants['placement'].mean(),
            'avg_week1_score': non_us_contestants['week1_avg_score'].mean(),
            'win_count': (non_us_contestants['placement'] == 1).sum()
        }
        
        # t检验
        if len(us_contestants) > 10 and len(non_us_contestants) > 10:
            t_stat, p_value = stats.ttest_ind(
                us_contestants['placement'].dropna(),
                non_us_contestants['placement'].dropna()
            )
            results['t_test_placement'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return results
    
    def build_predictive_model(self) -> dict:
        """
        构建预测模型分析各因素的综合影响
        
        Returns:
            模型结果字典
        """
        results = {}
        
        # 准备特征矩阵
        features = ['celebrity_age_during_season', 'industry_encoded', 
                    'partner_encoded', 'is_us', 'season']
        
        df_model = self.df.dropna(subset=features + ['placement'])
        df_model = df_model[(df_model['celebrity_age_during_season'] > 0) &
                            (df_model['week1_avg_score'] > 0)]
        
        if len(df_model) < 50:
            return {'error': 'Insufficient data for modeling'}
        
        X = df_model[features].values
        y = df_model['placement'].values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 线性回归
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        
        # 交叉验证
        cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='r2')
        
        results['linear_regression'] = {
            'r2_score': lr.score(X_scaled, y),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'coefficients': dict(zip(features, lr.coef_)),
            'intercept': lr.intercept_
        }
        
        # 逻辑回归（预测是否进入前3）
        y_top3 = (y <= 3).astype(int)
        
        logr = LogisticRegression(max_iter=1000)
        logr.fit(X_scaled, y_top3)
        
        cv_scores_log = cross_val_score(logr, X_scaled, y_top3, cv=5, scoring='accuracy')
        
        results['logistic_regression_top3'] = {
            'accuracy': logr.score(X_scaled, y_top3),
            'cv_accuracy_mean': cv_scores_log.mean(),
            'cv_accuracy_std': cv_scores_log.std(),
            'coefficients': dict(zip(features, logr.coef_[0])),
            'intercept': logr.intercept_[0]
        }
        
        return results
    
    def compare_judge_fan_impact(self, estimated_votes: dict = None) -> dict:
        """
        比较各因素对评委分数和观众投票的影响差异
        
        Args:
            estimated_votes: 估算的观众投票数据（可选）
        
        Returns:
            对比分析结果
        """
        results = {}
        
        # 分析各因素与评委分数的相关性
        df_analysis = self.df[(self.df['week1_avg_score'] > 0) & 
                              (self.df['celebrity_age_during_season'] > 0)].copy()
        
        # 年龄与评委分数
        corr_age_score, p_age_score = stats.pearsonr(
            df_analysis['celebrity_age_during_season'],
            df_analysis['week1_avg_score']
        )
        
        results['age_impact'] = {
            'correlation_with_judge_score': corr_age_score,
            'p_value': p_age_score,
            'interpretation': 'Older contestants tend to get ' + 
                            ('higher' if corr_age_score > 0 else 'lower') + ' scores'
        }
        
        # 行业与评委分数
        industry_scores = df_analysis.groupby('celebrity_industry')['week1_avg_score'].mean()
        results['industry_judge_scores'] = industry_scores.sort_values(ascending=False).to_dict()
        
        # 舞伴与评委分数
        partner_scores = df_analysis.groupby('ballroom_partner')['week1_avg_score'].mean()
        results['partner_judge_scores'] = partner_scores.sort_values(ascending=False).head(10).to_dict()
        
        return results


def generate_impact_report(df: pd.DataFrame) -> str:
    """
    生成影响因素分析报告
    
    Args:
        df: 清洗后的数据DataFrame
    
    Returns:
        报告文本
    """
    analyzer = ImpactAnalyzer(df)
    
    report = []
    report.append("=" * 60)
    report.append("IMPACT FACTOR ANALYSIS REPORT")
    report.append("Dancing with the Stars - MCM 2026 Problem C")
    report.append("=" * 60)
    
    # 1. 舞伴影响分析
    report.append("\n1. PROFESSIONAL PARTNER IMPACT")
    report.append("-" * 40)
    partner_results = analyzer.analyze_partner_impact()
    
    if 'anova_partner_placement' in partner_results:
        anova = partner_results['anova_partner_placement']
        report.append(f"ANOVA Test (Partner → Placement):")
        report.append(f"  F-statistic: {anova['f_statistic']:.4f}")
        report.append(f"  P-value: {anova['p_value']:.4f}")
        report.append(f"  Significant at α=0.05: {anova['significant']}")
    
    report.append("\nTop Partners by Win Rate:")
    for partner, rate in list(partner_results['win_rates'].items())[:5]:
        report.append(f"  {partner}: {rate:.2%}")
    
    # 2. 年龄影响分析
    report.append("\n2. AGE IMPACT")
    report.append("-" * 40)
    age_results = analyzer.analyze_age_impact()
    
    corr_data = age_results['correlations']
    report.append(f"Age vs Placement Correlation: {corr_data['age_vs_placement']['correlation']:.4f}")
    report.append(f"  P-value: {corr_data['age_vs_placement']['p_value']:.4f}")
    
    report.append("\nWinner Age Statistics:")
    winner_stats = age_results['winner_age_stats']
    report.append(f"  Mean: {winner_stats['mean']:.1f}")
    report.append(f"  Median: {winner_stats['median']:.1f}")
    report.append(f"  Range: {winner_stats['min']:.0f} - {winner_stats['max']:.0f}")
    
    # 3. 行业影响分析
    report.append("\n3. INDUSTRY IMPACT")
    report.append("-" * 40)
    industry_results = analyzer.analyze_industry_impact()
    
    if 'chi2_industry_top3' in industry_results:
        chi2 = industry_results['chi2_industry_top3']
        report.append(f"Chi-square Test (Industry → Top 3):")
        report.append(f"  Chi2: {chi2['chi2']:.4f}")
        report.append(f"  P-value: {chi2['p_value']:.4f}")
        report.append(f"  Significant at α=0.05: {chi2['significant']}")
    
    report.append("\nTop Industries by Win Rate:")
    for industry, rate in list(industry_results['industry_win_rates'].items())[:5]:
        report.append(f"  {industry}: {rate:.2%}")
    
    # 4. 预测模型
    report.append("\n4. PREDICTIVE MODEL ANALYSIS")
    report.append("-" * 40)
    model_results = analyzer.build_predictive_model()
    
    if 'error' not in model_results:
        lr = model_results['linear_regression']
        report.append(f"Linear Regression (Predicting Placement):")
        report.append(f"  R² Score: {lr['r2_score']:.4f}")
        report.append(f"  Cross-validation R²: {lr['cv_r2_mean']:.4f} ± {lr['cv_r2_std']:.4f}")
        report.append("  Feature Coefficients:")
        for feat, coef in lr['coefficients'].items():
            report.append(f"    {feat}: {coef:.4f}")
        
        logr = model_results['logistic_regression_top3']
        report.append(f"\nLogistic Regression (Predicting Top 3):")
        report.append(f"  Accuracy: {logr['accuracy']:.4f}")
        report.append(f"  Cross-validation Accuracy: {logr['cv_accuracy_mean']:.4f} ± {logr['cv_accuracy_std']:.4f}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_dwts_data, clean_data
    
    # 加载数据
    df = load_dwts_data()
    df = clean_data(df)
    
    # 生成分析报告
    report = generate_impact_report(df)
    print(report)
    
    # 保存报告
    output_path = Path(__file__).parent.parent / "reports"
    output_path.mkdir(exist_ok=True)
    with open(output_path / "impact_analysis_report.txt", "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path / 'impact_analysis_report.txt'}")
