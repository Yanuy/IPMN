import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
from typing import Dict
import json
import multiprocessing as mp
# 设置pandas多线程
import os
os.environ['NUMEXPR_MAX_THREADS'] = str(mp.cpu_count())

from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import os


class FraudDetectionSystem:
    """
    增强的金融欺诈检测系统
    支持多种算法，处理不平衡数据，生成图特征和时序特征
    整合了高效的滑动窗口算法，包含出度入度特征
    """

    def __init__(self, data_path: str = None, feature_cache_path: str = "features_cache.csv"):
        self.data_path = data_path
        self.feature_cache_path = feature_cache_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.feature_columns = []
        self.encoders = {}
        self.scalers = {}
        self.models = {}

        # 滑动窗口参数
        self.window_days = 30  # 默认30天窗口
        self.step_days = 7  # 默认7天步长

        # 配置参数
        self.train_ratio = 0.7
        self.sample_ratio = 1.0

    def load_data(self, sample_ratio: float = 1.0, chunk_size: int = 10000) -> pd.DataFrame:
        """
        高效加载大CSV文件
        """
        print(f"Loading data from {self.data_path}...")
        self.sample_ratio = sample_ratio

        try:
            if sample_ratio < 1.0:
                # 随机采样加载
                total_lines = sum(1 for _ in open(self.data_path, 'r', encoding='utf-8'))
                skip_idx = np.random.choice(range(1, total_lines),
                                            size=int((1 - sample_ratio) * total_lines),
                                            replace=False)
                self.df = pd.read_csv(self.data_path, skiprows=skip_idx)
            else:
                # 分块加载
                chunks = []
                for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                    chunks.append(chunk)
                self.df = pd.concat(chunks, ignore_index=True)

            print(f"Loaded {len(self.df)} transactions")
            print(f"Data shape: {self.df.shape}")

            # 数据预处理
            self._preprocess_data()
            return self.df

        except Exception as e:
            print(f"Error loading data: {e}")
            return self.df

    def _preprocess_data(self):
        """
        数据预处理
        """
        print("Preprocessing data...")

        # 转换日期时间
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')

        # 处理时间字段
        if 'Time' in self.df.columns:
            self.df['Hour'] = pd.to_datetime(self.df['Time'], format='%H:%M:%S', errors='coerce').dt.hour

        # 添加月份信息用于划分数据集
        self.df['Month'] = self.df['Date'].dt.to_period('M')

        # 排序
        self.df = self.df.sort_values(['Date', 'Time']).reset_index(drop=True)

        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"Fraud rate: {self.df['Is_laundering'].mean():.8f}")

    def split_data_by_month(self, train_ratio: float = 0.7):
        """
        按月份划分训练集和测试集
        """
        print(f"Splitting data with train ratio: {train_ratio}")

        self.train_ratio = train_ratio
        unique_months = sorted(self.df['Month'].unique())

        train_months_count = int(len(unique_months) * train_ratio)
        train_months = unique_months[:train_months_count]
        test_months = unique_months[train_months_count:]

        if test_months:
            first_test_month = test_months[0]
            test_start_date = pd.to_datetime(str(first_test_month)) + timedelta(days=7)

            self.train_df = self.df[self.df['Month'].isin(train_months)].copy()
            test_mask = (self.df['Month'].isin(test_months)) & (self.df['Date'] >= test_start_date)
            self.test_df = self.df[test_mask].copy()

            print(f"Training months: {len(train_months)} ({train_months[0]} to {train_months[-1]})")
            print(f"Testing months: {len(test_months)} ({test_months[0]} to {test_months[-1]})")
            print(f"Training samples: {len(self.train_df)}")
            print(f"Testing samples: {len(self.test_df)}")
            print(f"Training fraud rate: {self.train_df['Is_laundering'].mean():.8f}")
            print(f"Testing fraud rate: {self.test_df['Is_laundering'].mean():.8f}")
        else:
            raise ValueError("Not enough months for train/test split")

    def load_or_create_features(self, force_recreate: bool = False):
        """
        加载或创建特征
        """
        if os.path.exists(self.feature_cache_path) and not force_recreate:
            print(f"Loading cached features from {self.feature_cache_path}")
            try:
                cached_df = pd.read_csv(self.feature_cache_path)
                feature_cols = [col for col in cached_df.columns if col not in self.df.columns]
                if feature_cols:
                    self.df = pd.merge(self.df,
                                       cached_df[['Sender_account', 'Receiver_account', 'Date'] + feature_cols],
                                       on=['Sender_account', 'Receiver_account', 'Date'], how='left')
                    print(f"Loaded {len(feature_cols)} cached features")
                    return
            except Exception as e:
                print(f"Error loading cached features: {e}")

        print("Creating new features...")
        self._create_all_features()

        try:
            self.df.to_csv(self.feature_cache_path, index=False)
            print(f"Features cached to {self.feature_cache_path}")
        except Exception as e:
            print(f"Error caching features: {e}")

    def _create_all_features(self):
        """
        创建所有特征
        """
        try:
            # 基础特征
            self._create_basic_features()

            # 使用高效滑动窗口算法创建活动特征（包含出度入度）
            self._create_enhanced_activity_features()

            # 时间模式特征
            self._create_temporal_features()

            # 交易模式特征
            self._create_transaction_pattern_features()

            # 风险评分特征
            self._create_risk_score_features()

            print("All features created successfully")

        except Exception as e:
            print(f"Error in feature creation: {e}")

    def _create_basic_features(self):
        """
        创建基础特征
        """
        print("Creating basic features...")

        # 金额特征
        self.df['Amount_log'] = np.log1p(self.df['Amount'])
        self.df['Amount_sqrt'] = np.sqrt(self.df['Amount'])
        self.df['Amount_percentile'] = self.df['Amount'].rank(pct=True)

        # 整数金额标记
        self.df['Is_round_amount'] = (self.df['Amount'] % 1 == 0).astype(int)
        self.df['Is_large_round'] = ((self.df['Amount'] % 1000 == 0) & (self.df['Amount'] >= 1000)).astype(int)

        # 跨境标记
        self.df['Is_cross_border'] = (self.df['Sender_bank_location'] != self.df['Receiver_bank_location']).astype(int)

        # 货币不匹配标记
        self.df['Currency_mismatch'] = (self.df['Payment_currency'] != self.df['Received_currency']).astype(int)

        # 自转账标记
        self.df['Is_self_transfer'] = (self.df['Sender_account'] == self.df['Receiver_account']).astype(int)

    def _create_enhanced_activity_features(self):
        """
        使用高效滑动窗口算法创建增强的活动特征（包含出度入度），无数据泄露。
        """
        print("Creating enhanced activity features using sliding window...")

        # 定义将要创建的特征列
        self.feature_columns_list = [
            'Sender_send_amount', 'Sender_send_count', 'Sender_send_frequency',
            'Receiver_receive_amount', 'Receiver_receive_count', 'Receiver_receive_frequency',
            'Sender_receive_amount', 'Sender_receive_count', 'Sender_receive_frequency',
            'Receiver_send_amount', 'Receiver_send_count', 'Receiver_send_frequency',
            'Sender_out_degree', 'Sender_in_degree',
            'Receiver_in_degree', 'Receiver_out_degree',
            'Pair_transaction_count', 'Pair_transaction_amount',
            'Sender_total_activity_count', 'Receiver_total_activity_count'
        ]

        # 初始化特征列为 NaN
        for col in self.feature_columns_list:
            self.df[col] = np.nan

        # 调用核心滑动窗口计算逻辑
        self.df = self._window_slider(self.df, self.window_days, self.step_days)

        print("Enhanced activity features created successfully.")
        return self.df

    def _net_info_rti_with_degree(self, window_df: pd.DataFrame, window_days_for_freq_calc: int):
        """
        计算指定窗口内的 Sender、Receiver 的交易统计信息、度数特征和总活动计数。

        参数:
        - window_df: 窗口内的交易数据
        - window_days_for_freq_calc: 用于频率计算的天数

        返回:
        包含各种聚合统计DataFrame的字典
        """
        # 计算频率时使用的总天数
        total_days_for_freq = max(window_days_for_freq_calc, 1)  # 避免除零错误

        # ========== 修复：添加缺失的 sender 发送统计 ==========
        sender_send_stats = window_df.groupby("Sender_account").agg(
            Sender_send_amount_calc=("Amount", "sum"),
            Sender_send_count_calc=("Amount", "count")
        ).reset_index()
        sender_send_stats["Sender_send_frequency_calc"] = sender_send_stats[
                                                              "Sender_send_count_calc"] / total_days_for_freq

        # ========== Receiver 接收统计 ==========
        receiver_receive_stats = window_df.groupby("Receiver_account").agg(
            Receiver_receive_amount_calc=("Amount", "sum"),
            Receiver_receive_count_calc=("Amount", "count")
        ).reset_index()
        receiver_receive_stats["Receiver_receive_frequency_calc"] = receiver_receive_stats[
                                                                        "Receiver_receive_count_calc"] / total_days_for_freq

        # ========== Sender 作为接收者的统计 ==========
        sender_receive_stats = window_df.groupby("Receiver_account").agg(
            Sender_receive_amount_calc=("Amount", "sum"),
            Sender_receive_count_calc=("Amount", "count")
        ).reset_index()
        sender_receive_stats.rename(columns={"Receiver_account": "Sender_account"}, inplace=True)
        sender_receive_stats["Sender_receive_frequency_calc"] = sender_receive_stats[
                                                                    "Sender_receive_count_calc"] / total_days_for_freq

        # ========== Receiver 作为发送者的统计 ==========
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

        # ========== 账户对交易特征 ==========
        pair_stats = window_df.groupby(["Sender_account", "Receiver_account"]).agg(
            Pair_transaction_count_calc=("Amount", "count"),
            Pair_transaction_amount_calc=("Amount", "sum")
        ).reset_index()

        # ========== 总活动计数 ==========
        outgoing_counts_in_window = window_df['Sender_account'].value_counts()
        incoming_counts_in_window = window_df['Receiver_account'].value_counts()
        total_activity_counts_series = outgoing_counts_in_window.add(incoming_counts_in_window, fill_value=0)
        total_activity_counts_df = total_activity_counts_series.reset_index()
        total_activity_counts_df.columns = ['Account', 'Total_activity_count_calc']

        # 返回字典，索引已设置，方便后续 .map() 操作
        return {
            "sender_send_stats": sender_send_stats.set_index("Sender_account"),
            "receiver_receive_stats": receiver_receive_stats.set_index("Receiver_account"),
            "sender_receive_stats": sender_receive_stats.set_index("Sender_account"),
            "receiver_send_stats": receiver_send_stats.set_index("Receiver_account"),
            "sender_out_degree": sender_out_degree.set_index("Sender_account"),
            "receiver_in_degree": receiver_in_degree.set_index("Receiver_account"),
            "sender_in_degree": sender_in_degree.set_index("Sender_account"),
            "receiver_out_degree": receiver_out_degree.set_index("Receiver_account"),
            "pair_stats": pair_stats.set_index(["Sender_account", "Receiver_account"]),
            "total_activity_counts": total_activity_counts_df.set_index("Account")
        }

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

    def _window_slider(self, dataset: pd.DataFrame, window_days: int, step_days: int):
        """
        无数据泄露的滑动窗口核心实现。
        T日交易的特征，使用 [T-1-window_days+1, T-1] 区间的数据计算。

        参数:
        - dataset: 输入数据集
        - window_days: 历史窗口天数
        - step_days: 步长天数

        返回:
        添加了特征的数据集
        """
        print(
            f"Starting sliding window calculation with {window_days} days window and {step_days} days step (lagged features)...")

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
            # 特征滞后1天：目标时段T的特征，基于 T-1 及之前的数据
            # hist_window_end_date = target_period_start_date - pd.Timedelta(days=1)
            # hist_window_start_date = hist_window_end_date - pd.Timedelta(days=window_days - 1)
            # 特征不滞后：目标时段T的特征，基于当天及之前的数据
            hist_window_end_date = target_period_start_date
            hist_window_start_date = hist_window_end_date - pd.Timedelta(days=window_days-1)
            # 修改：调整历史窗口开始日期，确保不早于数据集的最小日期
            actual_hist_start_date = max(hist_window_start_date, min_dataset_date)

            # 计算实际可用的历史天数
            actual_window_days = (hist_window_end_date - actual_hist_start_date).days + 1

            # 验证时间窗口
            if not self._validate_time_window(actual_hist_start_date, hist_window_end_date, target_period_start_date):
                print(f"Warning: Time window validation failed for target {target_period_start_date.date()}")
                current_target_date_idx += step_days
                continue

            # 修改：显示实际使用的窗口信息
            if actual_window_days < window_days:
                print(
                    f"  Target period: {target_period_start_date.date()} to {target_period_end_date.date()}. Features from: {actual_hist_start_date.date()} to {hist_window_end_date.date()} (partial window: {actual_window_days} days)")
            else:
                print(
                    f"  Target period: {target_period_start_date.date()} to {target_period_end_date.date()}. Features from: {actual_hist_start_date.date()} to {hist_window_end_date.date()} (full window: {actual_window_days} days)")

            # 修改：删除原来的跳过逻辑
            # 原代码：
            # if hist_window_start_date < min_dataset_date:
            #     print(f"    Skipping target {target_period_start_date.date()}-{target_period_end_date.date()}: Not enough history for full window ({hist_window_start_date.date()} < {min_dataset_date.date()})")
            #     current_target_date_idx += step_days
            #     continue

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
                # 修改：使用实际的窗口天数进行频率计算
                aggregated_stats_map = self._net_info_rti_with_degree(
                    window_df_for_calc,
                    window_days_for_freq_calc=actual_window_days  # 使用实际天数而不是预设的window_days
                )
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

            # 定义特征映射规则
            feature_mappings = [
                # (account_column, stats_key, calc_column, target_column)
                ('Sender_account', 'sender_send_stats', 'Sender_send_amount_calc', 'Sender_send_amount'),
                ('Sender_account', 'sender_send_stats', 'Sender_send_count_calc', 'Sender_send_count'),
                ('Sender_account', 'sender_send_stats', 'Sender_send_frequency_calc', 'Sender_send_frequency'),

                ('Receiver_account', 'receiver_receive_stats', 'Receiver_receive_amount_calc',
                 'Receiver_receive_amount'),
                ('Receiver_account', 'receiver_receive_stats', 'Receiver_receive_count_calc', 'Receiver_receive_count'),
                ('Receiver_account', 'receiver_receive_stats', 'Receiver_receive_frequency_calc',
                 'Receiver_receive_frequency'),

                ('Sender_account', 'sender_receive_stats', 'Sender_receive_amount_calc', 'Sender_receive_amount'),
                ('Sender_account', 'sender_receive_stats', 'Sender_receive_count_calc', 'Sender_receive_count'),
                ('Sender_account', 'sender_receive_stats', 'Sender_receive_frequency_calc', 'Sender_receive_frequency'),

                ('Receiver_account', 'receiver_send_stats', 'Receiver_send_amount_calc', 'Receiver_send_amount'),
                ('Receiver_account', 'receiver_send_stats', 'Receiver_send_count_calc', 'Receiver_send_count'),
                ('Receiver_account', 'receiver_send_stats', 'Receiver_send_frequency_calc', 'Receiver_send_frequency'),

                ('Sender_account', 'sender_out_degree', 'Sender_out_degree_calc', 'Sender_out_degree'),
                ('Receiver_account', 'receiver_in_degree', 'Receiver_in_degree_calc', 'Receiver_in_degree'),
                ('Sender_account', 'sender_in_degree', 'Sender_in_degree_calc', 'Sender_in_degree'),
                ('Receiver_account', 'receiver_out_degree', 'Receiver_out_degree_calc', 'Receiver_out_degree'),

                ('Sender_account', 'total_activity_counts', 'Total_activity_count_calc', 'Sender_total_activity_count'),
                ('Receiver_account', 'total_activity_counts', 'Total_activity_count_calc',
                 'Receiver_total_activity_count'),
            ]

            # 使用安全映射函数
            self._safe_map_features(dataset, target_indices, aggregated_stats_map, feature_mappings)

            # 处理配对交易特征（需要特殊处理）
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
                except Exception as e:
                    print(f"Warning: Failed to map pair features: {e}")

            current_target_date_idx += step_days
            processed_windows += 1

        print(f"Processed {processed_windows} windows successfully.")

        # 填充 NaN 值
        for col in self.feature_columns_list:
            if col in dataset.columns:
                dataset[col] = dataset[col].fillna(0.0)

        return dataset

    def get_feature_summary(self):
        """返回特征工程的总结信息"""
        if hasattr(self, 'feature_columns_list'):
            feature_summary = {}
            for col in self.feature_columns_list:
                if col in self.df.columns:
                    feature_summary[col] = {
                        'non_null_count': self.df[col].count(),
                        'null_count': self.df[col].isnull().sum(),
                        'mean': self.df[col].mean(),
                        'std': self.df[col].std(),
                        'min': self.df[col].min(),
                        'max': self.df[col].max()
                    }
            return feature_summary
        return {}

    def _create_temporal_features(self):
        """
        创建时间模式特征
        """
        print("Creating temporal features...")

        # 小时特征
        if 'Hour' in self.df.columns:
            self.df['Is_business_hour'] = ((self.df['Hour'] >= 9) & (self.df['Hour'] <= 17)).astype(int)
            self.df['Is_night_hour'] = ((self.df['Hour'] >= 22) | (self.df['Hour'] <= 6)).astype(int)

        # 日期特征
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['Is_weekend'] = (self.df['DayOfWeek'] >= 5).astype(int)
        self.df['Day'] = self.df['Date'].dt.day
        self.df['Is_month_start'] = (self.df['Day'] <= 5).astype(int)
        self.df['Is_month_end'] = (self.df['Day'] >= 25).astype(int)

        # 月份和季度
        self.df['Month_num'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter

    def _create_transaction_pattern_features(self):
        """
        创建交易模式特征
        """
        print("Creating transaction pattern features...")

        # 编码分类变量
        categorical_cols = ['Payment_currency', 'Received_currency', 'Sender_bank_location',
                            'Receiver_bank_location', 'Payment_type']

        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le

        # # 支付类型频率特征(future)
        # if 'Payment_type' in self.df.columns:
        #     payment_counts = self.df['Payment_type'].value_counts()
        #     self.df['Payment_type_frequency'] = self.df['Payment_type'].map(payment_counts)

    def _create_risk_score_features(self):
        """
        创建基于规则的风险评分特征
        """
        print("Creating risk score features...")

        # 初始化风险评分
        self.df['Risk_score'] = 0

        # 大额交易风险


        # 跨境交易风险
        self.df['Risk_score'] += self.df['Is_cross_border'] * 1

        # 货币不匹配风险
        self.df['Risk_score'] += self.df['Currency_mismatch'] * 1

        # 非工作时间交易风险
        if 'Is_business_hour' in self.df.columns:
            self.df['Risk_score'] += (1 - self.df['Is_business_hour']) * 1

        # 高频交易风险
        if 'Sender_send_count' in self.df.columns:
            self.df['Risk_score'] += (self.df['Sender_send_count'] > 10).astype(int) * 2

        # 整数金额风险
        self.df['Risk_score'] += self.df['Is_large_round'] * 1

        # 现金交易风险
        if 'Payment_type' in self.df.columns:
            cash_types = ['Cash Deposit', 'Cash Withdrawal']
            self.df['Risk_score'] += self.df['Payment_type'].isin(cash_types).astype(int) * 1


    def set_window_parameters(self, window_days: int = 30, step_days: int = 7):
        """
        设置滑动窗口参数
        """
        self.window_days = window_days
        self.step_days = step_days
        print(f"Window parameters set: window_days={window_days}, step_days={step_days}")

    def create_advanced_degree_features(self):
        """
        创建高级度数特征
        """
        print("Creating advanced degree features...")

        # 度数比率特征
        if 'Sender_out_degree' in self.df.columns and 'Sender_in_degree' in self.df.columns:
            # 出入度比率
            self.df['Sender_degree_ratio'] = np.where(
                self.df['Sender_in_degree'] > 0,
                self.df['Sender_out_degree'] / self.df['Sender_in_degree'],
                self.df['Sender_out_degree']
            )

            self.df['Receiver_degree_ratio'] = np.where(
                self.df['Receiver_in_degree'] > 0,
                self.df['Receiver_out_degree'] / self.df['Receiver_in_degree'],
                self.df['Receiver_out_degree']
            )

        # 总度数
        if all(col in self.df.columns for col in ['Sender_out_degree', 'Sender_in_degree']):
            self.df['Sender_total_degree'] = self.df['Sender_out_degree'] + self.df['Sender_in_degree']
            self.df['Receiver_total_degree'] = self.df['Receiver_out_degree'] + self.df['Receiver_in_degree']

        # 账户对交易密度（交易金额/交易次数）
        if 'Pair_transaction_count' in self.df.columns and 'Pair_transaction_amount' in self.df.columns:
            self.df['Pair_avg_amount'] = np.where(
                self.df['Pair_transaction_count'] > 0,
                self.df['Pair_transaction_amount'] / self.df['Pair_transaction_count'],
                0
            )

        print("Advanced degree features created successfully")

    def get_degree_statistics(self):
        """
        获取度数特征的统计信息
        """
        degree_cols = ['Sender_out_degree', 'Sender_in_degree', 'Receiver_out_degree', 'Receiver_in_degree']
        existing_cols = [col for col in degree_cols if col in self.df.columns]

        if existing_cols:
            print("\n度数特征统计信息：")
            for col in existing_cols:
                print(f"{col}:")
                print(f"  平均值: {self.df[col].mean():.2f}")
                print(f"  最大值: {self.df[col].max()}")
                print(f"  95分位数: {self.df[col].quantile(0.95):.2f}")
                print(f"  标准差: {self.df[col].std():.2f}")

        if 'Pair_transaction_count' in self.df.columns:
            print(f"\n账户对交易统计：")
            print(f"  平均交易次数: {self.df['Pair_transaction_count'].mean():.2f}")
            print(f"  最大交易次数: {self.df['Pair_transaction_count'].max()}")
            print(f"  95分位数交易次数: {self.df['Pair_transaction_count'].quantile(0.95):.2f}")
#################################################################################################################

    def exploratory_data_analysis(self):
        """
        探索性数据分析和可视化
        """
        print("Performing exploratory data analysis...")

        try:
            # 创建图表目录
            os.makedirs('eda_plots', exist_ok=True)

            # 1. 基础统计
            self._plot_basic_statistics()

            # 2. 目标变量分布
            self._plot_target_distribution()

            # 3. 时间模式分析
            self._plot_temporal_patterns()

            # 4. 金额分布分析.l..
            self._plot_amount_distribution()

            # 5. 图特征分析
            self._plot_graph_features()

            # 6. 相关性分析
            self._plot_correlation_analysis()

            print("EDA completed. Plots saved in 'eda_plots' directory.")

        except Exception as e:
            print(f"Error in EDA: {e}")

    def _plot_basic_statistics(self):
        """
        基础统计图表
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 数据集大小
            axes[0, 0].bar(['Total', 'Fraud', 'Normal'],
                           [len(self.df), self.df['Is_laundering'].sum(),
                            len(self.df) - self.df['Is_laundering'].sum()])
            axes[0, 0].set_title('Dataset Composition')
            axes[0, 0].set_ylabel('Count')

            # 欺诈率
            fraud_rate = self.df['Is_laundering'].mean()
            axes[0, 1].pie([fraud_rate, 1 - fraud_rate], labels=['Fraud', 'Normal'], autopct='%1.2f%%')
            axes[0, 1].set_title('Fraud Rate')

            # 按月份的交易量
            monthly_counts = self.df.groupby(self.df['Date'].dt.to_period('M')).size()
            monthly_counts.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Monthly Transaction Volume')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 洗钱类型分布
            type_counts = self.df['Laundering_type'].value_counts().head(10)
            type_counts.plot(kind='barh', ax=axes[1, 1])
            axes[1, 1].set_title('Top 10 Laundering Types')

            plt.tight_layout()
            plt.savefig('eda_plots/basic_statistics.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting basic statistics: {e}")

    def _plot_target_distribution(self):
        """
        目标变量分布
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Is_laundering分布
            self.df['Is_laundering'].value_counts().plot(kind='bar', ax=axes[0])
            axes[0].set_title('Is_laundering Distribution')
            axes[0].set_xlabel('Is_laundering')
            axes[0].set_ylabel('Count')

            # Laundering_type分布（只显示前15个）
            top_types = self.df['Laundering_type'].value_counts().head(15)
            top_types.plot(kind='barh', ax=axes[1])
            axes[1].set_title('Top 15 Laundering Types')

            plt.tight_layout()
            plt.savefig('eda_plots/target_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting target distribution: {e}")

    def _plot_temporal_patterns(self):
        """
        时间模式分析
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 按小时的欺诈率
            if 'Hour' in self.df.columns:
                hourly_fraud = self.df.groupby('Hour')['Is_laundering'].agg(['count', 'sum', 'mean'])
                hourly_fraud['mean'].plot(kind='bar', ax=axes[0, 0])
                axes[0, 0].set_title('Fraud Rate by Hour')
                axes[0, 0].set_ylabel('Fraud Rate')

            # 按星期几的欺诈率
            if 'DayOfWeek' in self.df.columns:
                daily_fraud = self.df.groupby('DayOfWeek')['Is_laundering'].mean()
                daily_fraud.plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Fraud Rate by Day of Week')
                axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

            # 按月份的欺诈趋势
            monthly_fraud = self.df.groupby(self.df['Date'].dt.to_period('M'))['Is_laundering'].mean()
            monthly_fraud.plot(ax=axes[1, 0])
            axes[1, 0].set_title('Monthly Fraud Rate Trend')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 工作时间vs非工作时间
            if 'Is_business_hour' in self.df.columns:
                business_hour_fraud = self.df.groupby('Is_business_hour')['Is_laundering'].mean()
                business_hour_fraud.plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Fraud Rate: Business vs Non-Business Hours')
                axes[1, 1].set_xticklabels(['Non-Business', 'Business'])

            plt.tight_layout()
            plt.savefig('eda_plots/temporal_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting temporal patterns: {e}")

    def _plot_amount_distribution(self):
        """
        金额分布分析
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 金额分布（对数尺度）
            fraud_amounts = self.df[self.df['Is_laundering'] == 1]['Amount']
            normal_amounts = self.df[self.df['Is_laundering'] == 0]['Amount']

            axes[0, 0].hist([np.log1p(normal_amounts), np.log1p(fraud_amounts)],
                            bins=50, alpha=0.7, label=['Normal', 'Fraud'])
            axes[0, 0].set_title('Amount Distribution (Log Scale)')
            axes[0, 0].set_xlabel('Log(Amount + 1)')
            axes[0, 0].legend()

            # 金额箱线图
            amount_data = [normal_amounts.sample(min(10000, len(normal_amounts))),
                           fraud_amounts if len(fraud_amounts) > 0 else [0]]
            axes[0, 1].boxplot(amount_data, labels=['Normal', 'Fraud'])
            axes[0, 1].set_title('Amount Distribution by Fraud Status')
            axes[0, 1].set_yscale('log')

            # 整数金额分析
            if 'Is_round_amount' in self.df.columns:
                round_fraud = self.df.groupby('Is_round_amount')['Is_laundering'].mean()
                round_fraud.plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('Fraud Rate: Round vs Non-Round Amounts')
                axes[1, 0].set_xticklabels(['Non-Round', 'Round'])

            # 大额整数金额分析
            if 'Is_large_round' in self.df.columns:
                large_round_fraud = self.df.groupby('Is_large_round')['Is_laundering'].mean()
                large_round_fraud.plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Fraud Rate: Large Round Amounts')
                axes[1, 1].set_xticklabels(['Non-Large-Round', 'Large-Round'])

            plt.tight_layout()
            plt.savefig('eda_plots/amount_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting amount distribution: {e}")

    def _plot_graph_features(self):
        """
        图特征分析
        """
        try:
            graph_features = ['Sender_in_degree', 'Sender_out_degree', 'Sender_pagerank',
                              'Receiver_in_degree', 'Receiver_out_degree', 'Receiver_pagerank']

            available_features = [f for f in graph_features if f in self.df.columns]

            if not available_features:
                return

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, feature in enumerate(available_features[:6]):
                if feature in self.df.columns:
                    fraud_values = self.df[self.df['Is_laundering'] == 1][feature]
                    normal_values = self.df[self.df['Is_laundering'] == 0][feature]

                    # 采样以提高绘图效率
                    normal_sample = normal_values.sample(min(10000, len(normal_values)))

                    axes[i].hist([normal_sample, fraud_values], bins=50, alpha=0.7,
                                 label=['Normal', 'Fraud'], density=True)
                    axes[i].set_title(f'{feature} Distribution')
                    axes[i].legend()
                    axes[i].set_yscale('log')

            plt.tight_layout()
            plt.savefig('eda_plots/graph_features.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error plotting graph features: {e}")

    def _plot_correlation_analysis(self):
        """
        相关性分析
        """
        try:
            # 选择数值特征
            numeric_features = self.df.select_dtypes(include=[np.number]).columns
            numeric_features = [f for f in numeric_features if f not in ['Sender_account', 'Receiver_account']]

            if len(numeric_features) > 1:
                # 计算相关性矩阵
                corr_matrix = self.df[numeric_features].corr()

                # 与目标变量的相关性
                target_corr = corr_matrix['Is_laundering'].abs().sort_values(ascending=False)

                fig, axes = plt.subplots(1, 2, figsize=(20, 8))

                # 热图
                sns.heatmap(corr_matrix.iloc[:20, :20], annot=False, ax=axes[0], cmap='coolwarm')
                axes[0].set_title('Feature Correlation Matrix (Top 20)')

                # 与目标变量的相关性
                target_corr.head(20).plot(kind='barh', ax=axes[1])
                axes[1].set_title('Top 20 Features Correlated with Fraud')

                plt.tight_layout()
                plt.savefig('eda_plots/correlation_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"Error plotting correlation analysis: {e}")

    def prepare_features_for_modeling(self):
        """
        为建模准备特征
        """
        print("Preparing features for modeling...")

        # 选择特征列（排除目标变量和标识符）
        exclude_cols = ['Is_laundering', 'Laundering_type', 'Sender_account', 'Receiver_account',
                        'Date', 'Time', 'Month']

        self.feature_columns = [col for col in self.df.columns if col not in exclude_cols]

        print(f"Selected {len(self.feature_columns)} features for modeling")
        print("Feature columns:", self.feature_columns[:10], "..." if len(self.feature_columns) > 10 else "")

        # 处理缺失值
        for col in self.feature_columns:
            if self.df[col].dtype in ['object', 'category']:
                self.df[col] = self.df[col].fillna('unknown')
            else:
                self.df[col] = self.df[col].fillna(self.df[col].median())

    def train_models(self, task_mode=None):
        """
        训练多种模型 - 修正数据泄露问题并优化内存使用

        Parameters:
        -----------
        task_mode : str, optional
            指定训练任务类型:
            - 'Is_laundering' 或 'binary': 仅执行二分类任务
            - 'Laundering_type' 或 'multiclass': 仅执行多分类任务
            - None 或 'both': 执行所有任务（默认）
        """
        print("Training models...")

        if self.train_df is None or self.test_df is None:
            raise ValueError("Please split data first using split_data_by_month()")

        # 标准化任务模式参数
        if task_mode is not None:
            task_mode = task_mode.lower()
            if task_mode in ['is_laundering', 'binary', '二分类']:
                task_mode = 'binary'
            elif task_mode in ['laundering_type', 'multiclass', '多分类']:
                task_mode = 'multiclass'
            elif task_mode in ['both', 'all', '全部']:
                task_mode = None
            else:
                print(f"警告: 未识别的任务模式 '{task_mode}'，将执行所有任务")
                task_mode = None

        # 定义两个目标列
        target_columns = ['Is_laundering', 'Laundering_type']

        # 基础特征列清理 - 确保不包含任何目标列
        print("=== 检查并修复特征列 ===")
        print(f"原始特征列数量: {len(self.feature_columns)}")

        # 移除所有目标列from特征列
        base_feature_columns = [col for col in self.feature_columns if col not in target_columns]
        print(f"移除目标列后的基础特征列数量: {len(base_feature_columns)}")

        # 检查是否还有其他可能的泄露列
        potential_leakage_cols = []
        for col in base_feature_columns:
            if any(target in col.lower() for target in ['laundering', 'fraud', 'illegal', 'suspicious']):
                potential_leakage_cols.append(col)

        if potential_leakage_cols:
            print(f"警告: 发现可能的数据泄露列: {potential_leakage_cols}")
            response = input("是否移除这些列? (y/n): ")
            if response.lower() == 'y':
                base_feature_columns = [col for col in base_feature_columns if col not in potential_leakage_cols]
                print(f"移除可疑列后的基础特征列数量: {len(base_feature_columns)}")

        # 根据任务模式决定执行哪些任务
        execute_binary = task_mode is None or task_mode == 'binary'
        execute_multiclass = task_mode is None or task_mode == 'multiclass'

        print(f"\n=== 任务执行计划 ===")
        print(f"二分类任务: {'是' if execute_binary else '否'}")
        print(f"多分类任务: {'是' if execute_multiclass else '否'}")

        # 顺序执行任务以优化内存使用
        if execute_binary:
            self._execute_binary_classification(base_feature_columns)

        if execute_multiclass:
            self._execute_multiclass_classification(base_feature_columns)

    def _execute_binary_classification(self, base_feature_columns):
        """
        执行二分类任务 - 确保不使用多分类目标列作为特征
        """
        print("\n" + "=" * 50)
        print("开始执行二分类任务")
        print("=" * 50)

        # 为二分类任务准备特征列 - 明确排除多分类目标列
        binary_feature_columns = base_feature_columns.copy()

        # 双重检查：确保 Laundering_type 不在特征列中
        if 'Laundering_type' in binary_feature_columns:
            binary_feature_columns.remove('Laundering_type')
            print("警告: 从二分类特征中移除了 Laundering_type 列")

        print(f"二分类任务使用的特征列数量: {len(binary_feature_columns)}")
        print("=== 验证二分类特征列完整性 ===")
        print(f"Is_laundering 在特征列中: {'Is_laundering' in binary_feature_columns}")
        print(f"Laundering_type 在特征列中: {'Laundering_type' in binary_feature_columns}")

        # 准备二分类训练数据
        print("\n=== 准备二分类数据 ===")
        X_train_binary = self.train_df[binary_feature_columns].copy()
        y_train_binary = self.train_df['Is_laundering']

        X_test_binary = self.test_df[binary_feature_columns].copy()
        y_test_binary = self.test_df['Is_laundering']

        # 严格验证没有数据泄露
        assert 'Laundering_type' not in X_train_binary.columns, "数据泄露: 二分类训练数据包含多分类目标列 Laundering_type!"
        assert 'Is_laundering' not in X_train_binary.columns, "数据泄露: 二分类训练数据包含二分类目标列 Is_laundering!"

        print(f"二分类训练集形状: {X_train_binary.shape}")
        print(f"二分类目标分布: {y_train_binary.value_counts().to_dict()}")

        # 数据预处理
        print("\n=== 二分类数据预处理 ===")
        X_train_processed, X_test_processed = self._preprocess_features(
            X_train_binary, X_test_binary, 'binary'
        )

        # 训练模型
        print("\n=== 训练二分类模型 ===")
        self._train_binary_models(
            X_train_processed, y_train_binary,
            X_test_processed, y_test_binary
        )

        # 清理内存
        del X_train_binary, X_test_binary, X_train_processed, X_test_processed
        print("二分类任务完成，已清理相关数据")

    def _execute_multiclass_classification(self, base_feature_columns):
        """
        执行多分类任务 - 确保不使用二分类目标列作为特征
        """
        print("\n" + "=" * 50)
        print("开始执行多分类任务")
        print("=" * 50)

        # 为多分类任务准备特征列 - 明确排除二分类目标列
        multiclass_feature_columns = base_feature_columns.copy()

        # 双重检查：确保 Is_laundering 不在特征列中
        if 'Is_laundering' in multiclass_feature_columns:
            multiclass_feature_columns.remove('Is_laundering')
            print("警告: 从多分类特征中移除了 Is_laundering 列")

        print(f"多分类任务使用的特征列数量: {len(multiclass_feature_columns)}")
        print("=== 验证多分类特征列完整性 ===")
        print(f"Is_laundering 在特征列中: {'Is_laundering' in multiclass_feature_columns}")
        print(f"Laundering_type 在特征列中: {'Laundering_type' in multiclass_feature_columns}")

        # 准备多分类训练数据
        print("\n=== 准备多分类数据 ===")
        X_train_multi = self.train_df[multiclass_feature_columns].copy()
        y_train_multi = self.train_df['Laundering_type']

        X_test_multi = self.test_df[multiclass_feature_columns].copy()
        y_test_multi = self.test_df['Laundering_type']
        y_test_binary = self.test_df['Is_laundering']  # 用于评估欺诈检测性能

        # 严格验证没有数据泄露
        assert 'Is_laundering' not in X_train_multi.columns, "数据泄露: 多分类训练数据包含二分类目标列 Is_laundering!"
        assert 'Laundering_type' not in X_train_multi.columns, "数据泄露: 多分类训练数据包含多分类目标列 Laundering_type!"

        print(f"多分类训练集形状: {X_train_multi.shape}")
        print(f"多分类目标分布: {y_train_multi.value_counts().to_dict()}")

        # 数据预处理
        print("\n=== 多分类数据预处理 ===")
        X_train_processed, X_test_processed = self._preprocess_features(
            X_train_multi, X_test_multi, 'multi'
        )

        # 编码多分类目标
        le_multi = LabelEncoder()
        le_multi.fit(y_train_multi.astype(str))
        y_train_multi_encoded = le_multi.transform(y_train_multi.astype(str))

        # 处理测试集可能出现的新标签
        def safe_label_transform(encoder, labels):
            known_classes = set(encoder.classes_)
            return [encoder.transform([l])[0] if l in known_classes else -1 for l in labels.astype(str)]

        y_test_multi_encoded = safe_label_transform(le_multi, y_test_multi)
        self.encoders['laundering_type'] = le_multi

        # 打印多分类目标映射关系
        print("\n多分类目标映射:")
        mapping_multi = dict(zip(le_multi.classes_, le_multi.transform(le_multi.classes_)))
        for k, v in mapping_multi.items():
            print(f"  {k} -> {v}")

        # 训练模型
        print("\n=== 训练多分类模型 ===")
        self._train_multiclass_models(
            X_train_processed, y_train_multi_encoded,
            X_test_processed, y_test_multi_encoded, y_test_binary
        )

        # 清理内存
        del X_train_multi, X_test_multi, X_train_processed, X_test_processed
        del y_train_multi_encoded, y_test_multi_encoded
        print("多分类任务完成，已清理相关数据")

    def _validate_feature_integrity(self, feature_columns, task_type):
        """
        验证特征列的完整性，确保没有数据泄露
        """
        print(f"\n=== {task_type} 特征完整性验证 ===")

        if task_type == 'binary':
            forbidden_cols = ['Is_laundering', 'Laundering_type']
            print("二分类任务不应包含的列:", forbidden_cols)
        elif task_type == 'multiclass':
            forbidden_cols = ['Is_laundering', 'Laundering_type']
            print("多分类任务不应包含的列:", forbidden_cols)
        else:
            forbidden_cols = ['Is_laundering', 'Laundering_type']

        found_forbidden = [col for col in feature_columns if col in forbidden_cols]

        if found_forbidden:
            raise ValueError(f"数据泄露警告: {task_type} 任务的特征列中发现禁用列: {found_forbidden}")
        else:
            print(f"✓ {task_type} 特征列验证通过，未发现数据泄露")

        return True

    # 在每个任务执行前添加验证
    def _execute_binary_classification_with_validation(self, base_feature_columns):
        """
        执行二分类任务 - 带验证版本
        """
        # 准备二分类特征列
        binary_feature_columns = [col for col in base_feature_columns
                                  if col not in ['Is_laundering', 'Laundering_type']]

        # 验证特征完整性
        self._validate_feature_integrity(binary_feature_columns, 'binary')

        # 执行原有逻辑...
        self._execute_binary_classification(base_feature_columns)

    def _execute_multiclass_classification_with_validation(self, base_feature_columns):
        """
        执行多分类任务 - 带验证版本
        """
        # 准备多分类特征列
        multiclass_feature_columns = [col for col in base_feature_columns
                                      if col not in ['Is_laundering', 'Laundering_type']]

        # 验证特征完整性
        self._validate_feature_integrity(multiclass_feature_columns, 'multiclass')

        # 执行原有逻辑...
        self._execute_multiclass_classification(base_feature_columns)

    def _preprocess_features(self, X_train, X_test, task_type):
        """
        特征预处理 - 内存优化版本
        """
        print(f"开始 {task_type} 特征预处理...")

        # 就地修改以节省内存
        X_train_proc = X_train
        X_test_proc = X_test

        # 将非数值型特征转换为数值型
        encoder_key = f'{task_type}_encoders'
        if encoder_key not in self.encoders:
            self.encoders[encoder_key] = {}

        categorical_columns = []
        for col in X_train_proc.columns:
            if X_train_proc[col].dtype == 'object' or X_train_proc[col].dtype.name == 'category':
                categorical_columns.append(col)

        # 批量处理分类特征以节省内存
        for col in categorical_columns:
            print(f"编码列: {col}")
            le = LabelEncoder()

            # 使用更内存友好的方式处理
            train_values = X_train_proc[col].astype(str)
            test_values = X_test_proc[col].astype(str)

            # 获取唯一值
            all_values = pd.concat([train_values, test_values]).unique()
            le.fit(all_values)

            # 就地转换
            X_train_proc[col] = le.transform(train_values)
            X_test_proc[col] = le.transform(test_values)

            self.encoders[encoder_key][col] = le

        # 特征缩放
        scaler_key = f'{task_type}_scaler'
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_proc)
        X_test_scaled = scaler.transform(X_test_proc)
        self.scalers[scaler_key] = scaler

        print(f"{task_type} 特征预处理完成")
        print(f"  训练集形状: {X_train_scaled.shape}")
        print(f"  测试集形状: {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled

    def _train_binary_models(self, X_train, y_train, X_test, y_test):
        """
        训练二分类模型 - 内存优化版本
        """
        print("\n=== Training Binary Classification Models ===")

        # 计算类权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # 只保留必要的模型以节省内存
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=class_weights[1] / class_weights[0],
                random_state=42,
                eval_metric='logloss'
            )
        }

        # 可选：根据需要添加更多模型
        # 用户可以通过设置类属性来控制使用哪些模型
        # if hasattr(self, 'use_all_binary_models') and self.use_all_binary_models:
        #     models.update({
        #         'LightGBM': lgb.LGBMClassifier(
        #             n_estimators=100,
        #             max_depth=6,
        #             learning_rate=0.1,
        #             class_weight='balanced',
        #             random_state=42,
        #             verbose=-1
        #         ),
        #         'RandomForest': RandomForestClassifier(
        #             n_estimators=100,
        #             max_depth=10,
        #             class_weight='balanced',
        #             random_state=42,
        #             n_jobs=-1
        #         ),
        #         'LogisticRegression': LogisticRegression(
        #             class_weight='balanced',
        #             random_state=42,
        #             max_iter=1000
        #         )
        #     })

        if 'binary' not in self.models:
            self.models['binary'] = {}

        # 顺序训练模型以节省内存
        for name, model in models.items():
            print(f"\nTraining {name}...")

            try:
                # 根据模型类型决定是否使用SMOTE
                if name in ['XGBoost', 'LightGBM', 'RandomForest']:
                    model.fit(X_train, y_train)
                else:
                    # 为线性模型使用SMOTE，但立即清理
                    smote = SMOTE(random_state=42, k_neighbors=3)
                    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                    model.fit(X_train_smote, y_train_smote)
                    del X_train_smote, y_train_smote, smote  # 立即清理

                # 预测
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # 评估
                auc_score = roc_auc_score(y_test, y_pred_proba)

                print(f"{name} AUC: {auc_score:.8f}")
                print(f"{name} Classification Report:")
                print(classification_report(y_test, y_pred))

                # 保存模型和结果
                self.models['binary'][name] = {
                    'model': model,
                    'auc': auc_score,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }

            except Exception as e:
                print(f"Error training {name}: {e}")

    def _train_multiclass_models(self, X_train, y_train, X_test, y_test, y_test_binary):
        """
        训练多分类模型 - 内存优化版本
        """
        print("\n=== Training Multi-class Classification Models ===")

        models = {
            'XGBoost_Multi': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
        }

        if 'multiclass' not in self.models:
            self.models['multiclass'] = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            try:
                model.fit(X_train, y_train)

                # 预测
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # 多分类准确率
                accuracy = (y_pred == y_test).mean()

                # 欺诈检测准确率（不考虑具体类型）
                le = self.encoders['laundering_type']
                fraud_types = [i for i, label in enumerate(le.classes_) if not label.startswith('Normal')]

                y_pred_binary = np.isin(y_pred, fraud_types).astype(int)
                fraud_detection_accuracy = (y_pred_binary == y_test_binary).mean()
                fraud_auc = roc_auc_score(y_test_binary, np.max(y_pred_proba[:, fraud_types], axis=1) if len(
                    fraud_types) > 0 else y_pred_binary)

                print(f"{name} Multi-class Accuracy: {accuracy:.8f}")
                print(f"{name} Fraud Detection Accuracy: {fraud_detection_accuracy:.8f}")
                print(f"{name} Fraud Detection AUC: {fraud_auc:.8f}")

                # 保存模型和结果
                self.models['multiclass'][name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'fraud_detection_accuracy': fraud_detection_accuracy,
                    'fraud_auc': fraud_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'true_labels': y_test,
                    'class_names': le.classes_
                }

            except Exception as e:
                print(f"Error training {name}: {e}")

    def evaluate_models(self):
        """
        评估模型性能
        """
        print("\n=== Model Evaluation ===")

        try:
            # 创建评估图表目录
            os.makedirs('evaluation_plots', exist_ok=True)
            os.makedirs('enhanced_evaluation', exist_ok=True)
            print("Created directories: evaluation_plots, enhanced_evaluation")

            # 评估二分类模型（修复版）
            self._evaluate_binary_models_fixed()

            # 评估多分类模型（增强版）
            self._evaluate_multiclass_models_enhanced()

            # 生成详细的分类报告
            self._generate_detailed_classification_report()

            # 生成评估报告
            self._generate_evaluation_report_fixed()

            print(
                "Model evaluation completed. Plots saved in 'evaluation_plots' and 'enhanced_evaluation' directories.")

        except Exception as e:
            print(f"Error in model evaluation: {e}")
            import traceback
            traceback.print_exc()

    def _evaluate_binary_models_fixed(self):
        """
        评估二分类模型（修复版）
        """
        if 'binary' not in self.models or not self.models['binary']:
            print("No binary models to evaluate")
            return

        print("Evaluating binary models...")

        try:
            y_test = self.test_df['Is_laundering']

            # ROC曲线
            plt.figure(figsize=(12, 8))

            for name, model_info in self.models['binary'].items():
                try:
                    y_pred_proba = model_info['probabilities']
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    auc_score = model_info['auc']
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
                except Exception as e:
                    print(f"Error plotting ROC for {name}: {e}")

            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Binary Classification')
            plt.legend()
            plt.grid(True)
            plt.savefig('evaluation_plots/binary_roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Binary ROC curves saved")

            # 混淆矩阵（修复版）
            n_models = len(self.models['binary'])
            cols = min(2, n_models)
            rows = (n_models + 1) // 2

            fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            else:
                axes = axes.flatten()

            for i, (name, model_info) in enumerate(self.models['binary'].items()):
                if i >= len(axes):
                    break

                try:
                    y_pred = model_info['predictions']
                    cm = confusion_matrix(y_test, y_pred)

                    # 使用matplotlib的imshow而不是seaborn
                    im = axes[i].imshow(cm, interpolation='nearest', cmap='Blues')
                    axes[i].figure.colorbar(im, ax=axes[i])

                    # 添加文本注释
                    thresh = cm.max() / 2.
                    for row in range(cm.shape[0]):
                        for col in range(cm.shape[1]):
                            axes[i].text(col, row, format(cm[row, col], 'd'),
                                         ha="center", va="center",
                                         color="white" if cm[row, col] > thresh else "black")

                    axes[i].set_title(f'{name} Confusion Matrix')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Actual')
                    axes[i].set_xticks([0, 1])
                    axes[i].set_yticks([0, 1])
                    axes[i].set_xticklabels(['Normal', 'Fraud'])
                    axes[i].set_yticklabels(['Normal', 'Fraud'])

                except Exception as e:
                    print(f"Error plotting confusion matrix for {name}: {e}")

            # 隐藏多余的子图
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig('evaluation_plots/binary_confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Binary confusion matrices saved")

        except Exception as e:
            print(f"Error in binary model evaluation: {e}")
            import traceback
            traceback.print_exc()

    def _evaluate_multiclass_models_enhanced(self):
        """
        增强版多分类模型评估（修复版）
        """
        if 'multiclass' not in self.models or not self.models['multiclass']:
            print("No multiclass models to evaluate")
            return

        print("=== Enhanced Multi-class Model Evaluation ===")

        for model_name, model_info in self.models['multiclass'].items():
            print(f"\nEvaluating {model_name}...")

            try:
                y_true = model_info['true_labels']
                y_pred = model_info['predictions']
                y_pred_proba = model_info['probabilities']
                class_names = model_info['class_names']

                # 1. 详细混淆矩阵和分类报告
                self._plot_detailed_confusion_matrix_fixed(y_true, y_pred, class_names, model_name)

                # 2. 每个类别的性能分析
                self._analyze_class_performance(y_true, y_pred, y_pred_proba, class_names, model_name)

                # 3. ROC曲线（多分类）
                self._plot_multiclass_roc_curves(y_true, y_pred_proba, class_names, model_name)

                # 4. 预测概率分布分析（修复版）
                self._analyze_prediction_distribution_fixed(y_true, y_pred_proba, class_names, model_name)

                # 5. 错误分析
                self._analyze_classification_errors(y_true, y_pred, y_pred_proba, class_names, model_name)

                print(f"Completed evaluation for {model_name}")

            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                import traceback
                traceback.print_exc()

    def _plot_detailed_confusion_matrix_fixed(self, y_true, y_pred, class_names, model_name):
        """
        绘制详细的混淆矩阵（修复版）
        """
        try:
            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # 创建子图
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            # 原始计数矩阵 - 使用matplotlib直接绘制
            im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[0].figure.colorbar(im1, ax=axes[0], label='Count')

            # 添加文本注释
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[0].text(j, i, format(cm[i, j], 'd'),
                                 ha="center", va="center",
                                 color="white" if cm[i, j] > thresh else "black")

            axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)')
            axes[0].set_xlabel('Predicted Label')
            axes[0].set_ylabel('True Label')
            axes[0].set_xticks(range(len(class_names)))
            axes[0].set_yticks(range(len(class_names)))
            axes[0].set_xticklabels(class_names, rotation=45)
            axes[0].set_yticklabels(class_names)

            # 归一化矩阵
            im2 = axes[1].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            axes[1].figure.colorbar(im2, ax=axes[1], label='Proportion')

            # 添加文本注释
            thresh = cm_normalized.max() / 2.
            for i in range(cm_normalized.shape[0]):
                for j in range(cm_normalized.shape[1]):
                    axes[1].text(j, i, format(cm_normalized[i, j], '.3f'),
                                 ha="center", va="center",
                                 color="white" if cm_normalized[i, j] > thresh else "black")

            axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)')
            axes[1].set_xlabel('Predicted Label')
            axes[1].set_ylabel('True Label')
            axes[1].set_xticks(range(len(class_names)))
            axes[1].set_yticks(range(len(class_names)))
            axes[1].set_xticklabels(class_names, rotation=45)
            axes[1].set_yticklabels(class_names)

            plt.tight_layout()
            plt.savefig(f'enhanced_evaluation/{model_name}_detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Detailed confusion matrix saved for {model_name}")

        except Exception as e:
            print(f"Error plotting detailed confusion matrix for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_class_performance(self, y_true, y_pred, y_pred_proba, class_names, model_name):
        """
        分析每个类别的详细性能
        """
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score

            # 计算每个类别的指标
            precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

            # 计算每个类别的样本数量
            class_counts = np.bincount(y_true, minlength=len(class_names))

            # 创建性能表格
            performance_df = pd.DataFrame({
                'Class': class_names,
                'Sample_Count': class_counts,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Support': class_counts
            })

            # 保存详细报告
            performance_df.to_csv(f'enhanced_evaluation/{model_name}_class_performance.csv', index=False)

            # 可视化类别性能
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # 精确度
            axes[0, 0].bar(range(len(class_names)), precision, color='skyblue')
            axes[0, 0].set_title('Precision by Class')
            axes[0, 0].set_xlabel('Class')
            axes[0, 0].set_ylabel('Precision')
            axes[0, 0].set_xticks(range(len(class_names)))
            axes[0, 0].set_xticklabels(class_names, rotation=45)

            # 召回率
            axes[0, 1].bar(range(len(class_names)), recall, color='lightcoral')
            axes[0, 1].set_title('Recall by Class')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].set_xticks(range(len(class_names)))
            axes[0, 1].set_xticklabels(class_names, rotation=45)

            # F1分数
            axes[1, 0].bar(range(len(class_names)), f1, color='lightgreen')
            axes[1, 0].set_title('F1 Score by Class')
            axes[1, 0].set_xlabel('Class')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_xticks(range(len(class_names)))
            axes[1, 0].set_xticklabels(class_names, rotation=45)

            # 样本数量
            axes[1, 1].bar(range(len(class_names)), class_counts, color='gold')
            axes[1, 1].set_title('Sample Count by Class')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_xticks(range(len(class_names)))
            axes[1, 1].set_xticklabels(class_names, rotation=45)

            plt.tight_layout()
            plt.savefig(f'enhanced_evaluation/{model_name}_class_performance.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Class performance analysis saved for {model_name}")
            print("Performance summary:")
            print(performance_df.round(4))

        except Exception as e:
            print(f"Error in class performance analysis for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def _plot_multiclass_roc_curves(self, y_true, y_pred_proba, class_names, model_name):
        """
        绘制多分类ROC曲线
        """
        try:
            from sklearn.preprocessing import label_binarize
            from itertools import cycle

            # 二值化标签
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
            n_classes = len(class_names)

            # 如果只有两个类别，需要特殊处理
            if n_classes == 2:
                y_true_bin = np.column_stack([1 - y_true_bin.ravel(), y_true_bin.ravel()])

            # 计算每个类别的ROC曲线
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                if y_pred_proba.shape[1] > i:
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

            # 计算micro-average ROC
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # 绘制ROC曲线
            plt.figure(figsize=(12, 8))
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])

            for i, color in zip(range(n_classes), colors):
                if i in roc_auc:
                    plt.plot(fpr[i], tpr[i], color=color, lw=2,
                             label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

            plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
                     label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} - Multi-class ROC Curves')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(f'enhanced_evaluation/{model_name}_multiclass_roc.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Multi-class ROC curves saved for {model_name}")

        except Exception as e:
            print(f"Error plotting multi-class ROC for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_prediction_distribution_fixed(self, y_true, y_pred_proba, class_names, model_name):
        """
        分析预测概率分布（修复版）
        """
        try:
            n_classes = len(class_names)

            # 创建子图：每个类别的预测概率分布
            cols = min(3, n_classes)
            rows = (n_classes + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))

            # 处理单行或单列的情况
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            else:
                axes = axes.flatten()

            for i in range(n_classes):
                if i >= len(axes):
                    break

                # 获取该类别的真实样本和预测概率
                true_class_mask = (y_true == i)
                false_class_mask = (y_true != i)

                true_probs = y_pred_proba[true_class_mask, i]
                false_probs = y_pred_proba[false_class_mask, i]

                # 绘制概率分布 - 修复颜色问题
                if len(false_probs) > 0:
                    # 确保false_probs是一维数组
                    false_probs_flat = np.array(false_probs).flatten()
                    axes[i].hist(false_probs_flat, bins=30, alpha=0.7,
                                 label=f'Other classes (n={len(false_probs_flat)})',
                                 color='lightcoral', density=True)

                if len(true_probs) > 0:
                    # 确保true_probs是一维数组
                    true_probs_flat = np.array(true_probs).flatten()
                    axes[i].hist(true_probs_flat, bins=30, alpha=0.7,
                                 label=f'True {class_names[i]} (n={len(true_probs_flat)})',
                                 color='skyblue', density=True)

                axes[i].set_title(f'Prediction Probability Distribution\n{class_names[i]}')
                axes[i].set_xlabel('Predicted Probability')
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

            # 隐藏多余的子图
            for i in range(n_classes, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(f'enhanced_evaluation/{model_name}_prediction_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Prediction distribution analysis saved for {model_name}")

        except Exception as e:
            print(f"Error in prediction distribution analysis for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def _analyze_classification_errors(self, y_true, y_pred, y_pred_proba, class_names, model_name):
        """
        分析分类错误
        """
        try:
            # 找出错误分类的样本
            error_mask = (y_true != y_pred)
            error_indices = np.where(error_mask)[0]

            if len(error_indices) == 0:
                print(f"No classification errors found for {model_name}")
                return

            # 错误统计
            error_stats = []
            for true_class in range(len(class_names)):
                for pred_class in range(len(class_names)):
                    if true_class != pred_class:
                        mask = (y_true == true_class) & (y_pred == pred_class)
                        count = np.sum(mask)
                        if count > 0:
                            # 计算这些错误样本的平均置信度
                            avg_confidence = np.mean(np.max(y_pred_proba[mask], axis=1))
                            error_stats.append({
                                'True_Class': class_names[true_class],
                                'Predicted_Class': class_names[pred_class],
                                'Error_Count': count,
                                'Avg_Confidence': avg_confidence
                            })

            # 创建错误分析DataFrame
            error_df = pd.DataFrame(error_stats)
            if not error_df.empty:
                error_df = error_df.sort_values('Error_Count', ascending=False)
                error_df.to_csv(f'enhanced_evaluation/{model_name}_error_analysis.csv', index=False)

                # 可视化最常见的错误
                top_errors = error_df.head(10)

                if len(top_errors) > 0:
                    plt.figure(figsize=(12, 8))
                    error_labels = [f"{row['True_Class']} → {row['Predicted_Class']}"
                                    for _, row in top_errors.iterrows()]

                    plt.barh(range(len(top_errors)), top_errors['Error_Count'], color='lightcoral')
                    plt.yticks(range(len(top_errors)), error_labels)
                    plt.xlabel('Error Count')
                    plt.title(f'{model_name} - Most Common Classification Errors')
                    plt.grid(True, alpha=0.3)

                    # 添加置信度信息
                    for i, (_, row) in enumerate(top_errors.iterrows()):
                        plt.text(row['Error_Count'] + 0.1, i, f"Conf: {row['Avg_Confidence']:.3f}",
                                 va='center', fontsize=9)

                    plt.tight_layout()
                    plt.savefig(f'enhanced_evaluation/{model_name}_common_errors.png', dpi=300, bbox_inches='tight')
                    plt.close()

                    print(f"Error analysis completed for {model_name}. Top errors:")
                    print(top_errors.head())

        except Exception as e:
            print(f"Error in classification errors analysis for {model_name}: {e}")
            import traceback
            traceback.print_exc()

    def _generate_detailed_classification_report(self):
        """
        生成详细的分类报告
        """
        if 'multiclass' not in self.models or not self.models['multiclass']:
            return

        print("\n=== Generating Detailed Classification Report ===")

        try:
            for model_name, model_info in self.models['multiclass'].items():
                y_true = model_info['true_labels']
                y_pred = model_info['predictions']
                class_names = model_info['class_names']

                # 生成sklearn分类报告
                report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True,
                                               zero_division=0)

                # 转换为DataFrame并保存
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(f'enhanced_evaluation/{model_name}_classification_report.csv')

                # 打印摘要
                print(f"\n{model_name} Classification Report Summary:")
                print(f"Overall Accuracy: {report['accuracy']:.8f}")
                print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.8f}")
                print(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.8f}")

        except Exception as e:
            print(f"Error generating classification report: {e}")
            import traceback
            traceback.print_exc()

    def _generate_evaluation_report_fixed(self):
        """生成增强版评估报告（修复版）"""
        print("\n=== 生成增强版评估报告 ===")

        try:
            report_content = []
            report_content.append("# 洗钱检测模型评估报告（增强版）\n")
            report_content.append(f"生成时间: {pd.Timestamp.now()}\n\n")

            # 数据集概况
            report_content.append("## 数据集概况\n")
            if hasattr(self, 'test_df'):
                report_content.append(f"- 测试集样本数: {len(self.test_df)}\n")
                report_content.append(f"- 洗钱样本数: {self.test_df['Is_laundering'].sum()}\n")
                report_content.append(f"- 正常样本数: {(~self.test_df['Is_laundering']).sum()}\n")
                report_content.append(f"- 洗钱类型数: {self.test_df['Laundering_type'].nunique()}\n\n")

            # 模型性能总结
            if 'binary' in self.models and self.models['binary']:
                report_content.append("## 二分类模型性能\n")
                for name, model_info in self.models['binary'].items():
                    report_content.append(f"### {name}\n")
                    # 安全获取accuracy，如果不存在则使用其他指标计算
                    if 'accuracy' in model_info:
                        accuracy = model_info['accuracy']
                    else:
                        # 从预测结果计算准确率
                        if 'predictions' in model_info and hasattr(self, 'test_df'):
                            y_true = self.test_df['Is_laundering']
                            y_pred = model_info['predictions']
                            accuracy = (y_pred == y_true).mean()
                        else:
                            accuracy = 0.0

                    auc_score = model_info.get('auc', 0.0)
                    report_content.append(f"- 准确率: {accuracy:.8f}\n")
                    report_content.append(f"- AUC: {auc_score:.8f}\n\n")

            if 'multiclass' in self.models and self.models['multiclass']:
                report_content.append("## 多分类模型性能\n")
                for name, model_info in self.models['multiclass'].items():
                    report_content.append(f"### {name}\n")
                    report_content.append(f"- 多分类准确率: {model_info.get('accuracy', 0.0):.8f}\n")
                    report_content.append(f"- 欺诈检测准确率: {model_info.get('fraud_detection_accuracy', 0.0):.8f}\n")
                    report_content.append(f"- 欺诈检测AUC: {model_info.get('fraud_auc', 0.0):.8f}\n")

                    # 添加类别详细信息
                    if 'class_names' in model_info:
                        report_content.append(f"- 分类类别: {', '.join(model_info['class_names'])}\n")
                    report_content.append("\n")

            # 增强功能说明
            report_content.append("## 增强功能\n")
            report_content.append("本次评估包含以下增强功能：\n")
            report_content.append("1. 详细的混淆矩阵分析（原始计数和归一化）\n")
            report_content.append("2. 每个分类的精确度、召回率、F1分数分析\n")
            report_content.append("3. 多分类ROC曲线和AUC分析\n")
            report_content.append("4. 预测概率分布分析\n")
            report_content.append("5. 分类错误模式分析\n")
            report_content.append("6. 详细的分类报告\n\n")

            report_content.append("## 文件说明\n")
            report_content.append("- `*_detailed_confusion_matrix.png`: 详细混淆矩阵\n")
            report_content.append("- `*_class_performance.png/csv`: 各类别性能分析\n")
            report_content.append("- `*_multiclass_roc.png`: 多分类ROC曲线\n")
            report_content.append("- `*_prediction_distribution.png`: 预测概率分布\n")
            report_content.append("- `*_common_errors.png`: 常见分类错误\n")
            report_content.append("- `*_classification_report.csv`: 详细分类报告\n")

            # 保存报告
            with open('enhanced_evaluation/evaluation_report.md', 'w', encoding='utf-8') as f:
                f.writelines(report_content)

            print("增强版评估报告已生成: enhanced_evaluation/evaluation_report.md")

        except Exception as e:
            print(f"Error generating evaluation report: {e}")
            import traceback
            traceback.print_exc()

    def save_models(self, path: str = "models"):
        """
        保存训练好的模型
        """
        print(f"Saving models to {path}...")

        os.makedirs(path, exist_ok=True)

        try:
            # 保存模型
            for model_type, models in self.models.items():
                for name, model_info in models.items():
                    model_path = os.path.join(path, f"{model_type}_{name}.joblib")
                    joblib.dump(model_info['model'], model_path)

            # 保存编码器和缩放器
            joblib.dump(self.encoders, os.path.join(path, "encoders.joblib"))
            joblib.dump(self.scalers, os.path.join(path, "scalers.joblib"))

            # 保存特征列表
            with open(os.path.join(path, "feature_columns.json"), 'w') as f:
                json.dump(self.feature_columns, f)

            print("Models saved successfully!")

        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self, path: str = "models"):
        """
        加载训练好的模型
        """
        print(f"Loading models from {path}...")

        try:
            # 加载编码器和缩放器
            self.encoders = joblib.load(os.path.join(path, "encoders.joblib"))
            self.scalers = joblib.load(os.path.join(path, "scalers.joblib"))

            # 加载特征列表
            with open(os.path.join(path, "feature_columns.json"), 'r') as f:
                self.feature_columns = json.load(f)

            # 加载模型
            self.models = {'binary': {}, 'multiclass': {}}

            for file in os.listdir(path):
                if file.endswith('.joblib') and file not in ['encoders.joblib', 'scalers.joblib']:
                    model_type, name = file.replace('.joblib', '').split('_', 1)
                    model = joblib.load(os.path.join(path, file))
                    self.models[model_type][name] = {'model': model}

            print("Models loaded successfully!")

        except Exception as e:
            print(f"Error loading models: {e}")

    def predict(self, data: pd.DataFrame, model_name: str = 'XGBoost') -> Dict:
        """
        使用训练好的模型进行预测
        """
        print(f"Making predictions with {model_name}...")

        # 特征工程（简化版，只包含不依赖目标变量的特征）
        data_processed = data.copy()

        # 基础特征
        data_processed['Amount_log'] = np.log1p(data_processed['Amount'])
        data_processed['Is_cross_border'] = (
                    data_processed['Sender_bank_location'] != data_processed['Receiver_bank_location']).astype(int)
        data_processed['Currency_mismatch'] = (
                    data_processed['Payment_currency'] != data_processed['Received_currency']).astype(int)

        # 时间特征
        data_processed['Date'] = pd.to_datetime(data_processed['Date'])
        data_processed['Hour'] = pd.to_datetime(data_processed['Time'], format='%H:%M:%S').dt.hour
        data_processed['DayOfWeek'] = data_processed['Date'].dt.dayofweek
        data_processed['Is_weekend'] = (data_processed['DayOfWeek'] >= 5).astype(int)

        # 选择可用特征
        available_features = [col for col in self.feature_columns if col in data_processed.columns]
        X = data_processed[available_features]

        # 处理缺失值
        for col in available_features:
            if X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna('unknown')
            else:
                X[col] = X[col].fillna(X[col].median())

        # 特征缩放
        if 'standard' in self.scalers:
            X_scaled = self.scalers['standard'].transform(X)
        else:
            X_scaled = X

        results = {}

        # 二分类预测
        if 'binary' in self.models and model_name in self.models['binary']:
            model = self.models['binary'][model_name]['model']
            binary_pred = model.predict(X_scaled)
            binary_proba = model.predict_proba(X_scaled)[:, 1]

            results['is_laundering_prediction'] = binary_pred
            results['is_laundering_probability'] = binary_proba

        # 多分类预测
        if 'multiclass' in self.models and f'{model_name}_Multi' in self.models['multiclass']:
            model = self.models['multiclass'][f'{model_name}_Multi']['model']
            multi_pred = model.predict(X_scaled)
            multi_proba = model.predict_proba(X_scaled)

            # 解码预测结果
            le = self.encoders['laundering_type']
            type_pred = le.inverse_transform(multi_pred)

            results['laundering_type_prediction'] = type_pred
            results['laundering_type_probabilities'] = multi_proba

        return results

    def run_full_pipeline(self, data_path: str = None, sample_ratio: float = 1.0,
                          force_recreate_features: bool = False):
        """
        运行完整的流水线
        """
        print("=== Starting Fraud Detection System Pipeline ===")

        # 1. 加载数据
        if data_path:
            self.data_path = data_path
        self.load_data(sample_ratio=sample_ratio)

        # 2. 创建特征
        self.load_or_create_features(force_recreate=force_recreate_features)

        # 3. 数据探索
        #self.exploratory_data_analysis()

        # 4. 准备特征
        self.prepare_features_for_modeling()

        # 5. 划分数据
        self.split_data_by_month()

        # 6. 训练模型
        self.train_models()

        # 7. 评估模型
        self.evaluate_models()

        # 8. 保存模型
        self.save_models()

        print("=== Pipeline completed successfully! ===")


# 使用示例
if __name__ == "__main__":
    # 初始化系统
    fraud_system = FraudDetectionSystem()

    # 运行完整流水线
    # 如果有实际数据文件，替换为实际路径
    fraud_system.run_full_pipeline(
        data_path="SAML-D.csv",  # 替换为实际数据路径
        sample_ratio= 1,  # 使用10%的数据进行测试
        force_recreate_features=False
    )

    # 示例预测
    # sample_data = fraud_system.df.head(100).drop(['Is_laundering', 'Laundering_type'], axis=1)
    # predictions = fraud_system.predict(sample_data, model_name='XGBoost')
    # print("Sample predictions:", predictions)


