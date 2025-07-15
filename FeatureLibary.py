import re

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import patches

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import multiprocessing as mp
# 设置pandas多线程
import os
os.environ['NUMEXPR_MAX_THREADS'] = str(mp.cpu_count())
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import os
import pandas as pd
import igraph as ig
import plotly.graph_objects as go
import shap

def _create_all_features(self, feature_mode: str = 'full'):
    """
    创建所有特征

    Parameters:
    feature_mode: str
        - 'basic': 仅创建基础特征，不包含高级图特征
        - 'full': 创建所有特征（默认）
        - 'custom': 根据 self.feature_config 配置创建特征
    """
    try:
        # 根据模式设置特征配置
        if feature_mode == 'basic':
            # 基础模式：不包含高级图特征
            config = {
                'basic': True,
                'enhanced_activity': True,  # 滑动窗口特征，但不包含高级图
                'advanced_graph': False,  # 关闭滑动窗口中的高级图特征
                'temporal': True,
                'transaction_pattern': True,
                'risk_score': True,
                'advanced_graph_non_window': False,  # 关闭非滑动窗口的高级图特征
                'time_series': True,
                'behavioral_change': True,
                'currency_cross_border': True
            }
        elif feature_mode == 'full':
            # 全量模式：创建所有特征
            config = {
                'basic': True,
                'enhanced_activity': True,
                'advanced_graph': True,
                'temporal': True,
                'transaction_pattern': True,
                'risk_score': True,
                'advanced_graph_non_window': True,
                'time_series': True,
                'behavioral_change': True,
                'currency_cross_border': True
            }
        elif feature_mode == 'custom':
            # 自定义模式：使用实例的配置
            config = self.feature_config.copy()
        else:
            print(f"Warning: Unknown feature_mode '{feature_mode}', using 'full' mode")
            config = {
                'basic': True,
                'enhanced_activity': True,
                'advanced_graph': True,
                'temporal': True,
                'transaction_pattern': True,
                'risk_score': True,
                'advanced_graph_non_window': True,
                'time_series': True,
                'behavioral_change': True,
                'currency_cross_border': True
            }

        print(f"Feature creation mode: {feature_mode}")
        print(f"Feature configuration: {config}")

        # 基础特征
        if config.get('basic', False):
            print("Creating basic features...")
            self._create_basic_features()

        # 使用高效滑动窗口算法创建活动特征（包含出度入度）
        if config.get('enhanced_activity', False):
            print("Creating enhanced activity features...")
            # 传递高级图特征的开关给滑动窗口
            self._create_enhanced_activity_features(enable_advanced_graph=config.get('advanced_graph', False))

        # 时间模式特征
        if config.get('temporal', False):
            print("Creating temporal features...")
            self._create_temporal_features()

        # 交易模式特征
        if config.get('transaction_pattern', False):
            print("Creating transaction pattern features...")
            self._create_transaction_pattern_features()

        # 风险评分特征
        if config.get('risk_score', False):
            print("Creating risk score features...")
            self._create_risk_score_features()

        # 新增：增强的图特征（非滑动窗口）
        if config.get('advanced_graph_non_window', False):
            print("Creating advanced graph features (non-window)...")
            self._create_advanced_graph_features()

        # 新增：时间序列特征
        if config.get('time_series', False):
            print("Creating time series features...")
            self._create_time_series_features()

        # 新增：行为变化特征
        if config.get('behavioral_change', False):
            print("Creating behavioral change features...")
            self._create_behavioral_change_features()

        # 新增：多币种和跨境增强特征
        if config.get('currency_cross_border', False):
            print("Creating currency and cross-border features...")
            self._create_currency_and_cross_border_features()

        print("Feature creation completed successfully")

    except Exception as e:
        print(f"Error in feature creation: {e}")
        import traceback
        traceback.print_exc()


def _create_basic_features(self):
    """创建基础特征"""
    print("Creating basic features...")

    # 金额特征
    self.df['Amount_log'] = np.log1p(self.df['Amount'])
    self.df['Amount_sqrt'] = np.sqrt(self.df['Amount'])
    # 金额分箱特征
    self.df['Amount_bin'] = pd.cut(self.df['Amount'],
                                   bins=[0, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000,
                                         float('inf')],
                                   labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # 整数金额标记
    self.df['Is_round_amount'] = (self.df['Amount'] % 1 == 0).astype(int)
    self.df['Is_large_round'] = ((self.df['Amount'] % 1000 == 0) & (self.df['Amount'] >= 1000)).astype(int)

    # 新增：特殊金额模式
    self.df['Is_round_hundred'] = (self.df['Amount'] % 100 == 0).astype(int)
    self.df['Is_round_ten_thousand'] = (self.df['Amount'] % 10000 == 0).astype(int)
    self.df['Has_99_pattern'] = (self.df['Amount'] % 100 > 95).astype(int)  # 如999, 1999等

    # 跨境标记
    self.df['Is_cross_border'] = (self.df['Sender_bank_location'] != self.df['Receiver_bank_location']).astype(
        int)

    # 货币不匹配标记
    self.df['Currency_mismatch'] = (self.df['Payment_currency'] != self.df['Received_currency']).astype(int)

    # 自转账标记
    self.df['Is_self_transfer'] = (self.df['Sender_account'] == self.df['Receiver_account']).astype(int)


def _create_enhanced_activity_features(self, enable_advanced_graph: bool = True):
    """使用高效滑动窗口算法创建增强的活动特征"""
    print("Creating enhanced activity features using sliding window...")

    # 基础特征列
    basic_feature_columns = [
        # 原有特征
        'Sender_send_amount', 'Sender_send_count', 'Sender_send_frequency',
        'Receiver_receive_amount', 'Receiver_receive_count', 'Receiver_receive_frequency',
        'Sender_receive_amount', 'Sender_receive_count', 'Sender_receive_frequency',
        'Receiver_send_amount', 'Receiver_send_count', 'Receiver_send_frequency',
        'Sender_out_degree', 'Sender_in_degree',
        'Receiver_in_degree', 'Receiver_out_degree',
        'Pair_transaction_count', 'Pair_transaction_amount',
        'Sender_total_activity_count', 'Receiver_total_activity_count',
        # 统计特征
        'Sender_amount_variance', 'Receiver_amount_variance',
        'Sender_unique_receivers', 'Receiver_unique_senders',
        'Sender_avg_amount', 'Receiver_avg_amount',
        'Sender_amount_std', 'Receiver_amount_std',
        'Pair_avg_amount', 'Pair_time_span_days'
    ]

    # 高级图特征列
    advanced_graph_features = [
        'Sender_clustering_coef', 'Receiver_clustering_coef',
        'Sender_pagerank', 'Receiver_pagerank',
        'Has_2_cycle', 'Has_3_cycle',
        'Sender_betweenness_centrality', 'Receiver_betweenness_centrality'
    ]

    # 根据设置决定创建哪些特征
    if enable_advanced_graph:
        self.feature_columns_list = basic_feature_columns + advanced_graph_features
        print("Including advanced graph features in sliding window calculation")
    else:
        self.feature_columns_list = basic_feature_columns
        print("Excluding advanced graph features from sliding window calculation")

    # 初始化特征列为 NaN
    for col in self.feature_columns_list:
        self.df[col] = np.nan

    # 调用核心滑动窗口计算逻辑
    self.df = self._window_slider(self.df, self.window_days, self.step_days,
                                  enable_advanced_graph=enable_advanced_graph)

    print("Enhanced activity features created successfully.")
    return self.df


def _window_graph(self, window_df: pd.DataFrame, window_days_for_freq_calc: int, enable_advanced_graph: bool = True):
    """
    计算指定窗口内的 Sender、Receiver 的交易统计信息、度数特征、图特征和总活动计数。
    增强版本包含更多图论特征。
    """
    # 计算频率时使用的总天数
    total_days_for_freq = max(window_days_for_freq_calc, 1)  # 避免除零错误

    # ========== 原有统计 ==========
    # Sender 发送统计
    sender_send_stats = window_df.groupby("Sender_account").agg(
        Sender_send_amount_calc=("Amount", "sum"),
        Sender_send_count_calc=("Amount", "count"),
        Sender_amount_variance_calc=("Amount", "var"),
        Sender_avg_amount_calc=("Amount", "mean"),
        Sender_amount_std_calc=("Amount", "std")
    ).reset_index()
    sender_send_stats["Sender_send_frequency_calc"] = sender_send_stats[
                                                          "Sender_send_count_calc"] / total_days_for_freq

    # Receiver 接收统计
    receiver_receive_stats = window_df.groupby("Receiver_account").agg(
        Receiver_receive_amount_calc=("Amount", "sum"),
        Receiver_receive_count_calc=("Amount", "count"),
        Receiver_amount_variance_calc=("Amount", "var"),
        Receiver_avg_amount_calc=("Amount", "mean"),
        Receiver_amount_std_calc=("Amount", "std")
    ).reset_index()
    receiver_receive_stats["Receiver_receive_frequency_calc"] = receiver_receive_stats[
                                                                    "Receiver_receive_count_calc"] / total_days_for_freq

    # Sender 作为接收者的统计
    sender_receive_stats = window_df.groupby("Receiver_account").agg(
        Sender_receive_amount_calc=("Amount", "sum"),
        Sender_receive_count_calc=("Amount", "count")
    ).reset_index()
    sender_receive_stats.rename(columns={"Receiver_account": "Sender_account"}, inplace=True)
    sender_receive_stats["Sender_receive_frequency_calc"] = sender_receive_stats[
                                                                "Sender_receive_count_calc"] / total_days_for_freq

    # Receiver 作为发送者的统计
    receiver_send_stats = window_df.groupby("Sender_account").agg(
        Receiver_send_amount_calc=("Amount", "sum"),
        Receiver_send_count_calc=("Amount", "count")
    ).reset_index()
    receiver_send_stats.rename(columns={"Sender_account": "Receiver_account"}, inplace=True)
    receiver_send_stats["Receiver_send_frequency_calc"] = receiver_send_stats[
                                                              "Receiver_send_count_calc"] / total_days_for_freq

    # ========== 度数特征 ==========
    sender_out_degree = window_df.groupby("Sender_account")["Receiver_account"].nunique().reset_index()
    sender_out_degree.rename(columns={"Receiver_account": "Sender_out_degree_calc"}, inplace=True)

    receiver_in_degree = window_df.groupby("Receiver_account")["Sender_account"].nunique().reset_index()
    receiver_in_degree.rename(columns={"Sender_account": "Receiver_in_degree_calc"}, inplace=True)

    sender_in_degree = window_df.groupby("Receiver_account")["Sender_account"].nunique().reset_index()
    sender_in_degree.rename(
        columns={"Receiver_account": "Sender_account", "Sender_account": "Sender_in_degree_calc"}, inplace=True)

    receiver_out_degree = window_df.groupby("Sender_account")["Receiver_account"].nunique().reset_index()
    receiver_out_degree.rename(
        columns={"Sender_account": "Receiver_account", "Receiver_account": "Receiver_out_degree_calc"},
        inplace=True)

    # ========== 新增：唯一连接数 ==========
    sender_unique_receivers = window_df.groupby("Sender_account")["Receiver_account"].nunique().reset_index()
    sender_unique_receivers.rename(columns={"Receiver_account": "Sender_unique_receivers_calc"}, inplace=True)

    receiver_unique_senders = window_df.groupby("Receiver_account")["Sender_account"].nunique().reset_index()
    receiver_unique_senders.rename(columns={"Sender_account": "Receiver_unique_senders_calc"}, inplace=True)

    # ========== 账户对交易特征（增强版） ==========
    pair_stats = window_df.groupby(["Sender_account", "Receiver_account"]).agg(
        Pair_transaction_count_calc=("Amount", "count"),
        Pair_transaction_amount_calc=("Amount", "sum"),
        Pair_avg_amount_calc=("Amount", "mean"),
        Pair_first_date=("Date", "min"),
        Pair_last_date=("Date", "max")
    ).reset_index()
    # 计算账户对的时间跨度
    pair_stats["Pair_time_span_days_calc"] = (
            pair_stats["Pair_last_date"] - pair_stats["Pair_first_date"]).dt.days

    # ========== 总活动计数 ==========
    outgoing_counts_in_window = window_df['Sender_account'].value_counts()
    incoming_counts_in_window = window_df['Receiver_account'].value_counts()
    total_activity_counts_series = outgoing_counts_in_window.add(incoming_counts_in_window, fill_value=0)
    total_activity_counts_df = total_activity_counts_series.reset_index()
    total_activity_counts_df.columns = ['Account', 'Total_activity_count_calc']

    # ========== 新增：高级图特征 ==========
    if enable_advanced_graph:
        graph_features = self._calculate_graph_features(window_df)
    else:
        # 创建空的图特征字典以保持一致的返回结构
        graph_features = {
            'clustering_coef': pd.DataFrame(),
            'pagerank': pd.DataFrame(),
            'betweenness_centrality': pd.DataFrame(),
            'cycles': {
                'has_2_cycle': pd.DataFrame(),
                'has_3_cycle': pd.DataFrame()
            }
        }

    # 返回字典，索引已设置
    return {
        "sender_send_stats": sender_send_stats.set_index("Sender_account"),
        "receiver_receive_stats": receiver_receive_stats.set_index("Receiver_account"),
        "sender_receive_stats": sender_receive_stats.set_index("Sender_account"),
        "receiver_send_stats": receiver_send_stats.set_index("Receiver_account"),
        "sender_out_degree": sender_out_degree.set_index("Sender_account"),
        "receiver_in_degree": receiver_in_degree.set_index("Receiver_account"),
        "sender_in_degree": sender_in_degree.set_index("Sender_account"),
        "receiver_out_degree": receiver_out_degree.set_index("Receiver_account"),
        "sender_unique_receivers": sender_unique_receivers.set_index("Sender_account"),
        "receiver_unique_senders": receiver_unique_senders.set_index("Receiver_account"),
        "pair_stats": pair_stats.set_index(["Sender_account", "Receiver_account"]),
        "total_activity_counts": total_activity_counts_df.set_index("Account"),
        "graph_features": graph_features
    }


def _calculate_graph_features(self, window_df):
    """计算高级图特征 - 使用igraph代替networkx"""
    # 获取所有唯一的账户
    all_accounts = pd.concat([window_df['Sender_account'], window_df['Receiver_account']]).unique()
    account_to_idx = {acc: idx for idx, acc in enumerate(all_accounts)}
    idx_to_account = {idx: acc for acc, idx in account_to_idx.items()}

    # 准备边列表和权重
    edges = []
    weights = []
    edge_dict = {}

    for _, row in window_df.iterrows():
        src_idx = account_to_idx[row['Sender_account']]
        dst_idx = account_to_idx[row['Receiver_account']]
        edge_key = (src_idx, dst_idx)

        if edge_key in edge_dict:
            edge_dict[edge_key]['weight'] += row['Amount']
            edge_dict[edge_key]['count'] += 1
        else:
            edge_dict[edge_key] = {'weight': row['Amount'], 'count': 1}

    # 构建边列表和权重列表
    for (src, dst), attrs in edge_dict.items():
        edges.append((src, dst))
        weights.append(attrs['weight'])

    # 创建igraph有向图
    g = ig.Graph(n=len(all_accounts), edges=edges, directed=True)
    g.es['weight'] = weights

    # 计算各种图特征
    features = {}

    # 聚类系数（转换为无向图计算）
    g_undirected = g.as_undirected(mode="collapse", combine_edges="sum")
    clustering_values = g_undirected.transitivity_local_undirected(mode="zero")
    clustering_dict = {idx_to_account[i]: val for i, val in enumerate(clustering_values)}
    features['clustering_coef'] = pd.DataFrame.from_dict(clustering_dict, orient='index',
                                                         columns=['clustering_coef_calc'])

    # PageRank
    try:
        pagerank_values = g.pagerank(weights='weight')
        pagerank_dict = {idx_to_account[i]: val for i, val in enumerate(pagerank_values)}
        features['pagerank'] = pd.DataFrame.from_dict(pagerank_dict, orient='index', columns=['pagerank_calc'])
    except:
        features['pagerank'] = pd.DataFrame()

    # 介数中心性（对小图计算）
    if len(all_accounts) < 1000:  # 限制计算规模
        try:
            betweenness_values = g.betweenness(directed=True, weights='weight')
            # 归一化
            max_betweenness = max(betweenness_values) if betweenness_values else 1
            if max_betweenness > 0:
                betweenness_values = [v / max_betweenness for v in betweenness_values]
            betweenness_dict = {idx_to_account[i]: val for i, val in enumerate(betweenness_values)}
            features['betweenness_centrality'] = pd.DataFrame.from_dict(betweenness_dict, orient='index',
                                                                        columns=['betweenness_centrality_calc'])
        except:
            features['betweenness_centrality'] = pd.DataFrame()
    else:
        features['betweenness_centrality'] = pd.DataFrame()

    # 循环检测
    features['cycles'] = self._detect_cycles_igraph(g, idx_to_account)

    return features


def _detect_cycles_igraph(self, g, idx_to_account):
    """使用igraph检测图中的循环"""
    cycle_features = {}

    # 2-cycle检测（A->B->A）
    nodes_in_2cycle = set()

    # 检查每个节点的出邻居和入邻居
    for v in range(g.vcount()):
        out_neighbors = set(g.successors(v))
        in_neighbors = set(g.predecessors(v))

        # 检查是否存在既是出邻居又是入邻居的节点（形成2-cycle）
        mutual_neighbors = out_neighbors & in_neighbors
        if mutual_neighbors:
            nodes_in_2cycle.add(v)
            nodes_in_2cycle.update(mutual_neighbors)

    # 转换为账户名
    cycle_features['has_2_cycle'] = pd.DataFrame.from_dict(
        {idx_to_account[v]: 1 if v in nodes_in_2cycle else 0 for v in range(g.vcount())},
        orient='index', columns=['has_2_cycle_calc']
    )

    # 3-cycle检测（简化版）
    nodes_in_3cycle = set()

    # 对每个节点检查是否参与3-cycle
    for v in range(g.vcount()):
        out_neighbors = list(g.successors(v))

        # 检查v的出邻居之间是否有连接
        for i, n1 in enumerate(out_neighbors):
            for j, n2 in enumerate(out_neighbors):
                if i != j:
                    # 检查n1->n2或n2->n1是否存在
                    if g.are_connected(n1, n2) or g.are_connected(n2, n1):
                        # 再检查是否有边回到v
                        if g.are_connected(n2, v) or g.are_connected(n1, v):
                            nodes_in_3cycle.update([v, n1, n2])

    # 转换为账户名
    cycle_features['has_3_cycle'] = pd.DataFrame.from_dict(
        {idx_to_account[v]: 1 if v in nodes_in_3cycle else 0 for v in range(g.vcount())},
        orient='index', columns=['has_3_cycle_calc']
    )

    return cycle_features


def _window_slider(self, dataset: pd.DataFrame, window_days: int, step_days: int, enable_advanced_graph: bool = True):
    """无数据泄露的滑动窗口核心实现（增强版）"""
    print(f"Starting sliding window calculation with {window_days} days window and {step_days} days step...")

    if 'Date' not in dataset.columns or not pd.api.types.is_datetime64_any_dtype(dataset['Date']):
        print("Error: 'Date' column is missing or not in datetime format.")
        return dataset

    all_unique_dates = sorted(dataset['Date'].unique())
    if not all_unique_dates:
        print("No dates found in dataset. Exiting window slider.")
        return dataset

    min_dataset_date = all_unique_dates[0]

    # 预计算日期索引以提高性能
    date_to_idx = {date: idx for idx, date in enumerate(all_unique_dates)}

    current_target_date_idx = 0
    processed_windows = 0

    while current_target_date_idx < len(all_unique_dates):
        target_period_start_date = all_unique_dates[current_target_date_idx]
        target_period_end_date_idx = min(current_target_date_idx + step_days - 1, len(all_unique_dates) - 1)
        target_period_end_date = all_unique_dates[target_period_end_date_idx]

        # 定义历史数据计算窗口 [hist_start, hist_end]
        # 特征不滞后：目标时段T的特征，基于当天及之前的数据
        hist_window_end_date = target_period_start_date
        hist_window_start_date = hist_window_end_date - pd.Timedelta(days=window_days - 1)
        # 修改：调整历史窗口开始日期，确保不早于数据集的最小日期
        actual_hist_start_date = max(hist_window_start_date, min_dataset_date)

        # 计算实际可用的历史天数
        actual_window_days = (hist_window_end_date - actual_hist_start_date).days + 1

        # 验证时间窗口
        if not self._validate_time_window(actual_hist_start_date, hist_window_end_date,
                                          target_period_start_date):
            print(f"Warning: Time window validation failed for target {target_period_start_date.date()}")
            current_target_date_idx += step_days
            continue

        if actual_window_days < window_days:
            print(
                f"  Target period: {target_period_start_date.date()} to {target_period_end_date.date()}. Features from: {actual_hist_start_date.date()} to {hist_window_end_date.date()} (partial window: {actual_window_days} days)")
        else:
            print(
                f"  Target period: {target_period_start_date.date()} to {target_period_end_date.date()}. Features from: {actual_hist_start_date.date()} to {hist_window_end_date.date()} (full window: {actual_window_days} days)")

        # 使用调整后的历史窗口
        hist_data_mask = (dataset['Date'] >= actual_hist_start_date) & (dataset['Date'] <= hist_window_end_date)
        window_df_for_calc = dataset.loc[
            hist_data_mask, ['Date', 'Sender_account', 'Receiver_account', 'Amount']].copy()

        if window_df_for_calc.empty:
            print(
                f"    Skipping target {target_period_start_date.date()}-{target_period_end_date.date()}: No data in feature calculation window.")
            current_target_date_idx += step_days
            continue

        try:
            # 使用实际的窗口天数进行频率计算
            aggregated_stats_map = self._window_graph(window_df_for_calc,
                                                      window_days_for_freq_calc=actual_window_days,
                                                      enable_advanced_graph=enable_advanced_graph)
        except Exception as e:
            print(
                f"Error calculating features for window {actual_hist_start_date.date()}-{hist_window_end_date.date()}: {e}")
            current_target_date_idx += step_days
            continue

        target_rows_mask = (dataset['Date'] >= target_period_start_date) & (
                dataset['Date'] <= target_period_end_date)
        target_indices = dataset[target_rows_mask].index

        if target_indices.empty:
            current_target_date_idx += step_days
            continue

        # 定义特征映射规则（包含新增特征）
        feature_mappings = [
            # 原有特征
            ('Sender_account', 'sender_send_stats', 'Sender_send_amount_calc', 'Sender_send_amount'),
            ('Sender_account', 'sender_send_stats', 'Sender_send_count_calc', 'Sender_send_count'),
            ('Sender_account', 'sender_send_stats', 'Sender_send_frequency_calc', 'Sender_send_frequency'),
            ('Sender_account', 'sender_send_stats', 'Sender_amount_variance_calc', 'Sender_amount_variance'),
            ('Sender_account', 'sender_send_stats', 'Sender_avg_amount_calc', 'Sender_avg_amount'),
            ('Sender_account', 'sender_send_stats', 'Sender_amount_std_calc', 'Sender_amount_std'),

            ('Receiver_account', 'receiver_receive_stats', 'Receiver_receive_amount_calc',
             'Receiver_receive_amount'),
            ('Receiver_account', 'receiver_receive_stats', 'Receiver_receive_count_calc',
             'Receiver_receive_count'),
            ('Receiver_account', 'receiver_receive_stats', 'Receiver_receive_frequency_calc',
             'Receiver_receive_frequency'),
            ('Receiver_account', 'receiver_receive_stats', 'Receiver_amount_variance_calc',
             'Receiver_amount_variance'),
            ('Receiver_account', 'receiver_receive_stats', 'Receiver_avg_amount_calc', 'Receiver_avg_amount'),
            ('Receiver_account', 'receiver_receive_stats', 'Receiver_amount_std_calc', 'Receiver_amount_std'),

            ('Sender_account', 'sender_receive_stats', 'Sender_receive_amount_calc', 'Sender_receive_amount'),
            ('Sender_account', 'sender_receive_stats', 'Sender_receive_count_calc', 'Sender_receive_count'),
            ('Sender_account', 'sender_receive_stats', 'Sender_receive_frequency_calc',
             'Sender_receive_frequency'),

            ('Receiver_account', 'receiver_send_stats', 'Receiver_send_amount_calc', 'Receiver_send_amount'),
            ('Receiver_account', 'receiver_send_stats', 'Receiver_send_count_calc', 'Receiver_send_count'),
            ('Receiver_account', 'receiver_send_stats', 'Receiver_send_frequency_calc',
             'Receiver_send_frequency'),

            ('Sender_account', 'sender_out_degree', 'Sender_out_degree_calc', 'Sender_out_degree'),
            ('Receiver_account', 'receiver_in_degree', 'Receiver_in_degree_calc', 'Receiver_in_degree'),
            ('Sender_account', 'sender_in_degree', 'Sender_in_degree_calc', 'Sender_in_degree'),
            ('Receiver_account', 'receiver_out_degree', 'Receiver_out_degree_calc', 'Receiver_out_degree'),

            ('Sender_account', 'sender_unique_receivers', 'Sender_unique_receivers_calc',
             'Sender_unique_receivers'),
            ('Receiver_account', 'receiver_unique_senders', 'Receiver_unique_senders_calc',
             'Receiver_unique_senders'),

            ('Sender_account', 'total_activity_counts', 'Total_activity_count_calc',
             'Sender_total_activity_count'),
            ('Receiver_account', 'total_activity_counts', 'Total_activity_count_calc',
             'Receiver_total_activity_count'),
        ]

        # 使用安全映射函数
        self._safe_map_features(dataset, target_indices, aggregated_stats_map, feature_mappings)

        # 处理配对交易特征
        pair_stats = aggregated_stats_map.get("pair_stats", pd.DataFrame())
        if not pair_stats.empty:
            try:
                map_keys_pair = pd.MultiIndex.from_arrays([
                    dataset.loc[target_indices, 'Sender_account'],
                    dataset.loc[target_indices, 'Receiver_account']
                ])
                if 'Pair_transaction_count_calc' in pair_stats.columns:
                    dataset.loc[target_indices, 'Pair_transaction_count'] = map_keys_pair.map(
                        pair_stats['Pair_transaction_count_calc'])
                if 'Pair_transaction_amount_calc' in pair_stats.columns:
                    dataset.loc[target_indices, 'Pair_transaction_amount'] = map_keys_pair.map(
                        pair_stats['Pair_transaction_amount_calc'])
                if 'Pair_avg_amount_calc' in pair_stats.columns:
                    dataset.loc[target_indices, 'Pair_avg_amount'] = map_keys_pair.map(
                        pair_stats['Pair_avg_amount_calc'])
                if 'Pair_time_span_days_calc' in pair_stats.columns:
                    dataset.loc[target_indices, 'Pair_time_span_days'] = map_keys_pair.map(
                        pair_stats['Pair_time_span_days_calc'])
            except Exception as e:
                print(f"Warning: Failed to map pair features: {e}")

        # 处理图特征
        graph_features = aggregated_stats_map.get("graph_features", {})
        if graph_features:
            # 聚类系数
            if 'clustering_coef' in graph_features and not graph_features['clustering_coef'].empty:
                dataset.loc[target_indices, 'Sender_clustering_coef'] = dataset.loc[
                    target_indices, 'Sender_account'].map(
                    graph_features['clustering_coef']['clustering_coef_calc'])
                dataset.loc[target_indices, 'Receiver_clustering_coef'] = dataset.loc[
                    target_indices, 'Receiver_account'].map(
                    graph_features['clustering_coef']['clustering_coef_calc'])

            # PageRank
            if 'pagerank' in graph_features and not graph_features['pagerank'].empty:
                dataset.loc[target_indices, 'Sender_pagerank'] = dataset.loc[
                    target_indices, 'Sender_account'].map(graph_features['pagerank']['pagerank_calc'])
                dataset.loc[target_indices, 'Receiver_pagerank'] = dataset.loc[
                    target_indices, 'Receiver_account'].map(graph_features['pagerank']['pagerank_calc'])

            # 介数中心性
            if 'betweenness_centrality' in graph_features and not graph_features[
                'betweenness_centrality'].empty:
                dataset.loc[target_indices, 'Sender_betweenness_centrality'] = dataset.loc[
                    target_indices, 'Sender_account'].map(
                    graph_features['betweenness_centrality']['betweenness_centrality_calc'])
                dataset.loc[target_indices, 'Receiver_betweenness_centrality'] = dataset.loc[
                    target_indices, 'Receiver_account'].map(
                    graph_features['betweenness_centrality']['betweenness_centrality_calc'])

            # 循环特征
            if 'cycles' in graph_features:
                if 'has_2_cycle' in graph_features['cycles'] and not graph_features['cycles'][
                    'has_2_cycle'].empty:
                    dataset.loc[target_indices, 'Has_2_cycle'] = dataset.loc[
                        target_indices, 'Sender_account'].map(
                        graph_features['cycles']['has_2_cycle']['has_2_cycle_calc'])
                if 'has_3_cycle' in graph_features['cycles'] and not graph_features['cycles'][
                    'has_3_cycle'].empty:
                    dataset.loc[target_indices, 'Has_3_cycle'] = dataset.loc[
                        target_indices, 'Sender_account'].map(
                        graph_features['cycles']['has_3_cycle']['has_3_cycle_calc'])

        current_target_date_idx += step_days
        processed_windows += 1

    print(f"Processed {processed_windows} windows successfully.")

    # 填充 NaN 值
    for col in self.feature_columns_list:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(0.0)

    return dataset


def _create_advanced_graph_features(self):
    """创建高级图特征（非滑动窗口）"""
    print("Creating advanced graph features...")

    # 新增：二阶邻居特征（类似GNN的2-hop邻居）
    # 这些特征将在滑动窗口中计算

    # 新增：账户角色特征
    self.df['Sender_out_in_ratio'] = np.where(
        self.df['Sender_in_degree'] > 0,
        self.df['Sender_out_degree'] / self.df['Sender_in_degree'],
        self.df['Sender_out_degree']
    )

    self.df['Receiver_out_in_ratio'] = np.where(
        self.df['Receiver_in_degree'] > 0,
        self.df['Receiver_out_degree'] / self.df['Receiver_in_degree'],
        self.df['Receiver_out_degree']
    )

    # 新增：相对活跃度
    self.df['Sender_relative_activity'] = np.where(
        self.df['Sender_total_activity_count'].mean() > 0,
        self.df['Sender_total_activity_count'] / self.df['Sender_total_activity_count'].mean(),
        0
    )

    self.df['Receiver_relative_activity'] = np.where(
        self.df['Receiver_total_activity_count'].mean() > 0,
        self.df['Receiver_total_activity_count'] / self.df['Receiver_total_activity_count'].mean(),
        0
    )


def _create_time_series_features(self):
    """创建时间序列特征"""
    print("Creating time series features...")

    # 新增：交易时间间隔特征（需要在滑动窗口中计算）
    # 这里添加一些简单的时间特征

    # 时间戳转换为数值
    if 'Timestamp' in self.df.columns:
        self.df['Timestamp_numeric'] = self.df['Timestamp'].astype(np.int64) // 10 ** 9  # 转为秒

        # 计算与前一笔交易的时间差（同一账户）
        self.df = self.df.sort_values(['Sender_account', 'Timestamp'])
        self.df['Sender_time_since_last'] = self.df.groupby('Sender_account')['Timestamp_numeric'].diff()

        self.df = self.df.sort_values(['Receiver_account', 'Timestamp'])
        self.df['Receiver_time_since_last'] = self.df.groupby('Receiver_account')['Timestamp_numeric'].diff()

        # 恢复原排序
        self.df = self.df.sort_values(['Date', 'Time']).reset_index(drop=True)

        # 填充缺失值
        self.df['Sender_time_since_last'] = self.df['Sender_time_since_last'].fillna(86400)  # 默认1天
        self.df['Receiver_time_since_last'] = self.df['Receiver_time_since_last'].fillna(86400)


def _create_temporal_features(self):
    """创建时间模式特征（增强版）"""
    print("Creating temporal features...")

    # 小时特征
    if 'Hour' in self.df.columns:
        self.df['Is_business_hour'] = ((self.df['Hour'] >= 9) & (self.df['Hour'] <= 17)).astype(int)
        self.df['Is_night_hour'] = ((self.df['Hour'] >= 22) | (self.df['Hour'] <= 6)).astype(int)
        self.df['Is_lunch_hour'] = ((self.df['Hour'] >= 12) & (self.df['Hour'] <= 14)).astype(int)
        self.df['Is_early_morning'] = ((self.df['Hour'] >= 4) & (self.df['Hour'] <= 7)).astype(int)
        self.df['Hour_sin'] = np.sin(2 * np.pi * self.df['Hour'] / 24)
        self.df['Hour_cos'] = np.cos(2 * np.pi * self.df['Hour'] / 24)

        # 新增：小时分组
        self.df['Hour_group'] = pd.cut(self.df['Hour'],
                                       bins=[-1, 6, 12, 18, 24],
                                       labels=['night', 'morning', 'afternoon', 'evening'])
        self.df['Hour_group_encoded'] = LabelEncoder().fit_transform(self.df['Hour_group'])

    # 分钟和秒特征
    if 'Minute' in self.df.columns:
        self.df['Minute_sin'] = np.sin(2 * np.pi * self.df['Minute'] / 60)
        self.df['Minute_cos'] = np.cos(2 * np.pi * self.df['Minute'] / 60)
        self.df['Is_round_minute'] = (self.df['Minute'] % 10 == 0).astype(int)

    # 日期特征
    self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
    self.df['Is_weekend'] = (self.df['DayOfWeek'] >= 5).astype(int)
    self.df['Is_monday'] = (self.df['DayOfWeek'] == 0).astype(int)
    self.df['Is_friday'] = (self.df['DayOfWeek'] == 4).astype(int)
    self.df['Day'] = self.df['Date'].dt.day
    self.df['Is_month_start'] = (self.df['Day'] <= 5).astype(int)
    self.df['Is_month_end'] = (self.df['Day'] >= 25).astype(int)
    self.df['Is_mid_month'] = ((self.df['Day'] >= 14) & (self.df['Day'] <= 16)).astype(int)

    # 周期性特征
    self.df['Day_sin'] = np.sin(2 * np.pi * self.df['Day'] / 31)
    self.df['Day_cos'] = np.cos(2 * np.pi * self.df['Day'] / 31)
    self.df['Week_sin'] = np.sin(2 * np.pi * self.df['DayOfWeek'] / 7)
    self.df['Week_cos'] = np.cos(2 * np.pi * self.df['DayOfWeek'] / 7)

    # 月份和季度
    self.df['Month_num'] = self.df['Date'].dt.month
    self.df['Quarter'] = self.df['Date'].dt.quarter
    self.df['Month_sin'] = np.sin(2 * np.pi * self.df['Month_num'] / 12)
    self.df['Month_cos'] = np.cos(2 * np.pi * self.df['Month_num'] / 12)

    # 新增：年内天数
    self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
    self.df['DayOfYear_sin'] = np.sin(2 * np.pi * self.df['DayOfYear'] / 365)
    self.df['DayOfYear_cos'] = np.cos(2 * np.pi * self.df['DayOfYear'] / 365)

    # 新增：是否为节假日前后（简化版，可根据实际节假日调整）
    self.df['Is_near_holiday'] = (
            (self.df['Month_num'] == 12) & (self.df['Day'] >= 20) |  # 圣诞前
            (self.df['Month_num'] == 1) & (self.df['Day'] <= 10) |  # 新年后
            (self.df['Month_num'] == 7) & (self.df['Day'] <= 10)  # 假期季
    ).astype(int)


def _create_transaction_pattern_features(self):
    """创建交易模式特征（增强版）"""
    print("Creating transaction pattern features...")

    # 编码分类变量
    categorical_cols = ['Payment_currency', 'Received_currency', 'Sender_bank_location',
                        'Receiver_bank_location', 'Payment_type']

    for col in categorical_cols:
        if col in self.df.columns:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le

            # 新增：类别频率编码
            value_counts = self.df[col].value_counts()
            self.df[f'{col}_frequency'] = self.df[col].map(value_counts)
            self.df[f'{col}_frequency_log'] = np.log1p(self.df[f'{col}_frequency'])

    # 新增：支付类型组合特征
    if 'Payment_type' in self.df.columns:
        # 高风险支付类型
        high_risk_payment_types = ['Cash Deposit', 'Cash Withdrawal', 'Wire Transfer']
        self.df['Is_high_risk_payment'] = self.df['Payment_type'].isin(high_risk_payment_types).astype(int)

        # 新增：现金相关交易
        self.df['Is_cash_transaction'] = self.df['Payment_type'].str.contains('Cash', case=False,
                                                                              na=False).astype(int)

        # 新增：电子交易
        electronic_types = ['ACH', 'Wire Transfer', 'Credit Card', 'Debit Card']
        self.df['Is_electronic_payment'] = self.df['Payment_type'].isin(electronic_types).astype(int)


def _create_behavioral_change_features(self):
    """创建行为变化特征"""
    print("Creating behavioral change features...")

    # 新增：金额变化率
    if 'Sender_avg_amount' in self.df.columns:
        self.df['Sender_amount_deviation'] = np.where(
            self.df['Sender_avg_amount'] > 0,
            (self.df['Amount'] - self.df['Sender_avg_amount']) / self.df['Sender_avg_amount'],
            0
        )

        self.df['Receiver_amount_deviation'] = np.where(
            self.df['Receiver_avg_amount'] > 0,
            (self.df['Amount'] - self.df['Receiver_avg_amount']) / self.df['Receiver_avg_amount'],
            0
        )

        # 标记异常大额交易
        self.df['Is_sender_unusual_amount'] = (
                (self.df['Sender_amount_deviation'] > 2) |  # 超过平均值2倍
                (self.df['Amount'] > self.df['Sender_avg_amount'] + 2 * self.df['Sender_amount_std'])
        ).astype(int)

        self.df['Is_receiver_unusual_amount'] = (
                (self.df['Receiver_amount_deviation'] > 2) |
                (self.df['Amount'] > self.df['Receiver_avg_amount'] + 2 * self.df['Receiver_amount_std'])
        ).astype(int)

    # 新增：频率变化
    if 'Sender_send_frequency' in self.df.columns:
        # 计算历史平均频率（这是一个简化版本）
        self.df['Sender_frequency_change'] = np.where(
            self.df['Sender_send_frequency'].shift(1) > 0,
            (self.df['Sender_send_frequency'] - self.df['Sender_send_frequency'].shift(1)) / self.df[
                'Sender_send_frequency'].shift(1),
            0
        )

        self.df['Receiver_frequency_change'] = np.where(
            self.df['Receiver_receive_frequency'].shift(1) > 0,
            (self.df['Receiver_receive_frequency'] - self.df['Receiver_receive_frequency'].shift(1)) / self.df[
                'Receiver_receive_frequency'].shift(1),
            0
        )

    # 新增：新关系标记
    if 'Pair_transaction_count' in self.df.columns:
        self.df['Is_new_relationship'] = (self.df['Pair_transaction_count'] == 1).astype(int)
        self.df['Is_rare_relationship'] = (self.df['Pair_transaction_count'] <= 2).astype(int)


def _create_currency_and_cross_border_features(self):
    """创建多币种和跨境增强特征"""
    print("Creating currency and cross-border features...")

    # 新增：具体的跨境路径
    if 'Sender_bank_location' in self.df.columns and 'Receiver_bank_location' in self.df.columns:
        self.df['Cross_border_path'] = self.df['Sender_bank_location'] + '_to_' + self.df[
            'Receiver_bank_location']

        # 编码跨境路径
        le = LabelEncoder()
        self.df['Cross_border_path_encoded'] = le.fit_transform(self.df['Cross_border_path'])
        self.encoders['Cross_border_path'] = le

        # 高风险国家/地区标记（示例）
        high_risk_locations = ['UAE', 'Nigeria', 'Panama', 'Bahamas']  # 示例高风险地区
        self.df['Is_high_risk_sender_location'] = self.df['Sender_bank_location'].isin(
            high_risk_locations).astype(int)
        self.df['Is_high_risk_receiver_location'] = self.df['Receiver_bank_location'].isin(
            high_risk_locations).astype(int)
        self.df['Is_high_risk_transaction'] = (
                self.df['Is_high_risk_sender_location'] | self.df['Is_high_risk_receiver_location']
        ).astype(int)

    # 新增：货币对特征
    if 'Payment_currency' in self.df.columns and 'Received_currency' in self.df.columns:
        self.df['Currency_pair'] = self.df['Payment_currency'] + '_' + self.df['Received_currency']

        # 编码货币对
        le = LabelEncoder()
        self.df['Currency_pair_encoded'] = le.fit_transform(self.df['Currency_pair'])
        self.encoders['Currency_pair'] = le

        # 常见货币标记
        major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'UK pounds', 'US Dollar']
        self.df['Is_major_currency_payment'] = self.df['Payment_currency'].isin(major_currencies).astype(int)
        self.df['Is_major_currency_received'] = self.df['Received_currency'].isin(major_currencies).astype(int)

        # 新增：是否涉及加密货币或特殊货币（如果数据中有）
        crypto_currencies = ['Bitcoin', 'BTC', 'Ethereum', 'ETH', 'Crypto']  # 示例
        self.df['Is_crypto_involved'] = (
                self.df['Payment_currency'].isin(crypto_currencies) |
                self.df['Received_currency'].isin(crypto_currencies)
        ).astype(int)


def _create_risk_score_features(self):
    """创建基于规则的风险评分特征"""
    print("Creating risk score features...")

    # 初始化风险评分
    self.df['Risk_score'] = 0

    # 大额交易风险（分级）
    if 'Amount' in self.df.columns:
        amount_percentiles = self.df['Amount'].quantile([0.9, 0.95, 0.99])
        self.df['Risk_score'] += (self.df['Amount'] > amount_percentiles[0.9]).astype(int) * 1
        self.df['Risk_score'] += (self.df['Amount'] > amount_percentiles[0.95]).astype(int) * 2
        self.df['Risk_score'] += (self.df['Amount'] > amount_percentiles[0.99]).astype(int) * 3

    # 跨境交易风险
    self.df['Risk_score'] += self.df['Is_cross_border'] * 2

    # 货币不匹配风险
    self.df['Risk_score'] += self.df['Currency_mismatch'] * 2

    # 非工作时间交易风险
    if 'Is_business_hour' in self.df.columns:
        self.df['Risk_score'] += (1 - self.df['Is_business_hour']) * 1

    # 深夜交易额外风险
    if 'Is_night_hour' in self.df.columns:
        self.df['Risk_score'] += self.df['Is_night_hour'] * 2

    # 高频交易风险
    if 'Sender_send_count' in self.df.columns:
        self.df['Risk_score'] += (self.df['Sender_send_count'] > 10).astype(int) * 2
        self.df['Risk_score'] += (self.df['Sender_send_count'] > 50).astype(int) * 3

    # 整数金额风险
    self.df['Risk_score'] += self.df['Is_large_round'] * 2

    # 现金交易风险
    if 'Is_cash_transaction' in self.df.columns:
        self.df['Risk_score'] += self.df['Is_cash_transaction'] * 3

    # 新关系风险
    if 'Is_new_relationship' in self.df.columns:
        self.df['Risk_score'] += self.df['Is_new_relationship'] * 1

    # 异常金额风险
    if 'Is_sender_unusual_amount' in self.df.columns:
        self.df['Risk_score'] += self.df['Is_sender_unusual_amount'] * 3
        self.df['Risk_score'] += self.df['Is_receiver_unusual_amount'] * 3

    # 高风险地区
    if 'Is_high_risk_transaction' in self.df.columns:
        self.df['Risk_score'] += self.df['Is_high_risk_transaction'] * 4

    # 循环交易风险
    if 'Has_2_cycle' in self.df.columns:
        self.df['Risk_score'] += self.df['Has_2_cycle'] * 3
    if 'Has_3_cycle' in self.df.columns:
        self.df['Risk_score'] += self.df['Has_3_cycle'] * 2

    # 创建风险等级
    risk_thresholds = self.df['Risk_score'].quantile([0.7, 0.9, 0.95])
    self.df['Risk_level'] = pd.cut(
        self.df['Risk_score'],
        bins=[-np.inf, risk_thresholds[0.7], risk_thresholds[0.9], risk_thresholds[0.95], np.inf],
        labels=['low', 'medium', 'high', 'very_high']
    )
    self.df['Risk_level_encoded'] = LabelEncoder().fit_transform(self.df['Risk_level'])


def _validate_time_window(self, hist_start: pd.Timestamp, hist_end: pd.Timestamp,
                          target_start: pd.Timestamp) -> bool:
    """验证时间窗口是否符合无数据泄露要求"""
    return hist_end <= target_start


def _safe_map_features(self, dataset: pd.DataFrame, target_indices: pd.Index,
                       stats_dict: dict, feature_mappings: list):
    """安全地映射特征，处理缺失值和异常情况"""
    for mapping in feature_mappings:
        account_col, stats_key, calc_col, target_col = mapping

        if stats_key in stats_dict and not stats_dict[stats_key].empty:
            if calc_col in stats_dict[stats_key].columns:
                try:
                    mapped_values = dataset.loc[target_indices, account_col].map(
                        stats_dict[stats_key][calc_col]
                    )
                    dataset.loc[target_indices, target_col] = mapped_values
                except Exception as e:
                    print(f"Warning: Failed to map {target_col}: {e}")


def get_feature_summary(self):
    """返回特征工程的总结信息"""
    feature_summary = {}

    # 获取所有数值型特征
    numeric_features = self.df.select_dtypes(include=[np.number]).columns

    for col in numeric_features:
        if col not in ['Is_laundering', 'Laundering_type']:  # 排除目标变量
            feature_summary[col] = {
                'non_null_count': self.df[col].count(),
                'null_count': self.df[col].isnull().sum(),
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'unique_values': self.df[col].nunique()
            }

    return feature_summary
