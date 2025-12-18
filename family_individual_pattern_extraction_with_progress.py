"""
家庭和个人活动模式提取模块（带进度条版本）

在原有 family_individual_pattern_extraction_corrected.py 基础上
添加 tqdm 进度条支持，便于追踪长时间运行的任务进度

使用方法：
将此文件中的相关方法替换到原文件中，或直接使用此文件
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


# ==================== 配置类 ====================

@dataclass
class PatternExtractionConfig:
    """模式提取配置"""
    max_family_size: int = 8
    max_activities: int = 6
    person_n_components_range: Tuple[int, int] = (2, 15)
    household_n_components_range: Tuple[int, int] = (2, 10)
    covariance_type: str = 'full'
    random_state: int = 42
    
    # 特征配置
    continuous_dims: int = 2  # 开始时间, 结束时间
    purpose_classes: int = 10
    mode_classes: int = 11
    driver_classes: int = 2
    joint_classes: int = 2
    
    @property
    def activity_dim(self) -> int:
        return (self.continuous_dims + self.purpose_classes + 
                self.mode_classes + self.driver_classes + self.joint_classes)


# ==================== 特征提取（带进度条） ====================

class PersonFeatureExtractor:
    """个人特征提取器"""
    
    def __init__(self, config: PatternExtractionConfig):
        self.config = config
        
    def extract_features(
        self,
        person_id: str,
        household_id: str,
        activities: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """从单个人的活动序列提取统计特征"""
        valid = activities[mask.astype(bool)]
        n_activities = len(valid)
        
        if n_activities == 0:
            return np.zeros(self.get_feature_dim())
        
        # 时间特征
        start_times = valid[:, 0]
        end_times = valid[:, 1]
        
        # 目的分布
        purposes = valid[:, 2:12]
        purpose_dist = np.mean(purposes, axis=0)
        purpose_dist = purpose_dist / (np.sum(purpose_dist) + 1e-10)
        
        # 方式分布
        modes = valid[:, 12:23]
        mode_dist = np.mean(modes, axis=0)
        mode_dist = mode_dist / (np.sum(mode_dist) + 1e-10)
        
        # 联合出行和驾驶
        is_driver = valid[:, 23:25]
        is_joint = valid[:, 25:27]
        driver_ratio = np.mean(np.argmax(is_driver, axis=1))
        joint_ratio = np.mean(np.argmax(is_joint, axis=1))
        
        # 汇总特征
        features = np.concatenate([
            [n_activities],
            [np.mean(start_times), np.std(start_times) + 1e-10],
            [np.mean(end_times), np.std(end_times) + 1e-10],
            [np.mean(end_times - start_times)],  # 平均持续时间
            purpose_dist,
            mode_dist,
            [driver_ratio, joint_ratio]
        ])
        
        return features
    
    def get_feature_dim(self) -> int:
        """特征维度: 1 + 2 + 2 + 1 + 10 + 11 + 2 = 29"""
        return 29
    
    def get_feature_names(self) -> List[str]:
        return [
            'n_activities',
            'start_time_mean', 'start_time_std',
            'end_time_mean', 'end_time_std',
            'avg_duration',
            *[f'purpose_{i}' for i in range(10)],
            *[f'mode_{i}' for i in range(11)],
            'driver_ratio', 'joint_ratio'
        ]


class HouseholdFeatureExtractor:
    """家庭特征提取器"""
    
    def __init__(self, config: PatternExtractionConfig):
        self.config = config
        
    def extract_features(
        self,
        household_id: str,
        household_info: Dict,
        member_activities: List[np.ndarray],
        member_masks: List[np.ndarray]
    ) -> np.ndarray:
        """从家庭所有成员的活动提取家庭特征"""
        n_members = len(member_activities)
        
        # 收集所有活动
        all_activities = []
        for activities, mask in zip(member_activities, member_masks):
            valid = activities[mask.astype(bool)]
            if len(valid) > 0:
                all_activities.append(valid)
        
        if len(all_activities) == 0:
            return np.zeros(self.get_feature_dim())
        
        all_activities = np.vstack(all_activities)
        
        # 家庭基本信息
        n_workers = household_info.get('n_workers', 0)
        n_vehicles = household_info.get('n_vehicles', 0)
        
        # 时间特征
        start_times = all_activities[:, 0]
        end_times = all_activities[:, 1]
        
        # 车辆使用（mode 2,3 对应小汽车）
        modes = all_activities[:, 12:23]
        mode_indices = np.argmax(modes, axis=1)
        car_usage = np.mean((mode_indices == 2) | (mode_indices == 3))
        
        # 联合出行
        is_joint = all_activities[:, 25:27]
        joint_flags = np.argmax(is_joint, axis=1)
        joint_ratio = np.mean(joint_flags)
        
        # 目的分布
        purposes = all_activities[:, 2:12]
        purpose_dist = np.mean(purposes, axis=0)
        purpose_dist = purpose_dist / (np.sum(purpose_dist) + 1e-10)
        
        # 时段分布（早中晚夜）
        hours = start_times * 6 + 12  # 假设z-score转换
        hours = np.clip(hours, 0, 24)
        time_dist = np.zeros(4)
        time_dist[0] = np.mean((hours >= 6) & (hours < 10))
        time_dist[1] = np.mean((hours >= 10) & (hours < 16))
        time_dist[2] = np.mean((hours >= 16) & (hours < 20))
        time_dist[3] = np.mean((hours >= 20) | (hours < 6))
        
        features = np.concatenate([
            [n_members, n_workers, n_vehicles],
            [car_usage, joint_ratio],
            [np.min(start_times), np.max(end_times)],
            [np.max(end_times) - np.min(start_times)],  # 活动时间窗口
            purpose_dist,
            time_dist
        ])
        
        return features
    
    def get_feature_dim(self) -> int:
        """特征维度: 3 + 2 + 2 + 1 + 10 + 4 = 22"""
        return 22
    
    def get_feature_names(self) -> List[str]:
        return [
            'n_members', 'n_workers', 'n_vehicles',
            'car_usage_rate', 'joint_trip_ratio',
            'first_departure', 'last_return',
            'active_window',
            *[f'hh_purpose_{i}' for i in range(10)],
            'time_morning', 'time_midday', 'time_evening', 'time_night'
        ]


# ==================== GMM 聚类（带进度条） ====================

class PatternGMM:
    """通用GMM聚类器（带进度条）"""
    
    def __init__(
        self,
        n_components_range: Tuple[int, int] = (2, 15),
        covariance_type: str = 'full',
        random_state: int = 42,
        name: str = "GMM"
    ):
        self.n_components_range = n_components_range
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.name = name
        
        self.scaler = StandardScaler()
        self.gmm: Optional[GaussianMixture] = None
        self.best_n_components: Optional[int] = None
        
    def _select_n_components(
        self,
        X: np.ndarray,
        method: str = 'bic',
        show_progress: bool = True
    ) -> int:
        """选择最佳聚类数"""
        scores = []
        n_range = range(self.n_components_range[0], self.n_components_range[1] + 1)
        
        iterator = tqdm(n_range, desc=f"{self.name}: 选择最佳聚类数", disable=not show_progress)
        
        for n in iterator:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=3,
                max_iter=100
            )
            gmm.fit(X)
            
            if method == 'bic':
                score = gmm.bic(X)
            elif method == 'aic':
                score = gmm.aic(X)
            elif method == 'silhouette':
                labels = gmm.predict(X)
                if len(np.unique(labels)) > 1:
                    score = -silhouette_score(X, labels)
                else:
                    score = np.inf
            else:
                score = gmm.bic(X)
            
            scores.append(score)
            iterator.set_postfix({'n': n, 'score': f'{score:.2f}'})
        
        best_idx = np.argmin(scores)
        return list(n_range)[best_idx]
    
    def fit(
        self,
        X: np.ndarray,
        auto_select_k: bool = True,
        n_components: Optional[int] = None,
        show_progress: bool = True
    ) -> 'PatternGMM':
        """训练GMM"""
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 选择聚类数
        if auto_select_k:
            self.best_n_components = self._select_n_components(
                X_scaled, show_progress=show_progress
            )
            if show_progress:
                print(f"{self.name}: 选择 {self.best_n_components} 个聚类")
        else:
            self.best_n_components = n_components or self.n_components_range[0]
        
        # 训练最终模型
        if show_progress:
            print(f"{self.name}: 训练最终GMM模型...")
        
        self.gmm = GaussianMixture(
            n_components=self.best_n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5,
            max_iter=200
        )
        self.gmm.fit(X_scaled)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签"""
        X_scaled = self.scaler.transform(X)
        return self.gmm.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测聚类概率"""
        X_scaled = self.scaler.transform(X)
        return self.gmm.predict_proba(X_scaled)
    
    def get_cluster_centers(self) -> np.ndarray:
        """获取聚类中心（原始空间）"""
        return self.scaler.inverse_transform(self.gmm.means_)


# ==================== 活动模式提取器（带进度条） ====================

class ActivityPatternExtractor:
    """活动模式提取器（完整pipeline，带进度条）"""
    
    def __init__(self, config: PatternExtractionConfig = None):
        self.config = config or PatternExtractionConfig()
        
        self.person_extractor = PersonFeatureExtractor(self.config)
        self.household_extractor = HouseholdFeatureExtractor(self.config)
        
        self.person_gmm = PatternGMM(
            n_components_range=self.config.person_n_components_range,
            covariance_type=self.config.covariance_type,
            random_state=self.config.random_state,
            name="个人模式GMM"
        )
        
        self.household_gmm = PatternGMM(
            n_components_range=self.config.household_n_components_range,
            covariance_type=self.config.covariance_type,
            random_state=self.config.random_state,
            name="家庭模式GMM"
        )
        
        self.person_features_list: List[np.ndarray] = []
        self.household_features_list: List[np.ndarray] = []
        self.household_ids: List[str] = []
        
    def _prepare_data_from_raw(
        self,
        family_df: pd.DataFrame,
        member_df: pd.DataFrame,
        activity_df: pd.DataFrame,
        show_progress: bool = True
    ) -> Tuple[Dict[str, Dict], List[str]]:
        """从原始DataFrame准备数据结构"""
        
        household_data = {}
        household_ids = family_df['家庭编号'].unique().tolist()
        
        iterator = tqdm(household_ids, desc="准备数据结构", disable=not show_progress)
        
        for hh_id in iterator:
            # 家庭信息
            hh_info = family_df[family_df['家庭编号'] == hh_id].iloc[0]
            
            # 成员信息
            members = member_df[member_df['家庭编号'] == hh_id]
            
            # 活动信息
            hh_activities = activity_df[activity_df['家庭编号'] == hh_id]
            
            member_list = []
            for _, member in members.iterrows():
                member_id = member['成员编号']
                
                # 获取该成员的活动
                person_acts = hh_activities[hh_activities['家庭成员编号'] == member_id]
                
                # 转换为活动矩阵
                activities = np.zeros((self.config.max_activities, self.config.activity_dim))
                mask = np.zeros(self.config.max_activities)
                
                for idx, (_, act) in enumerate(person_acts.iterrows()):
                    if idx >= self.config.max_activities:
                        break
                    
                    # 时间特征（需要根据实际数据调整）
                    try:
                        start_time = float(act.get('出发时间1小时时间段', 12)) / 24 * 2 - 1
                        end_time = float(act.get('到达时间1小时时间段', 13)) / 24 * 2 - 1
                    except:
                        start_time, end_time = 0, 0.1
                    
                    activities[idx, 0] = start_time
                    activities[idx, 1] = end_time
                    
                    # 目的 one-hot
                    try:
                        purpose_idx = int(act.get('ActivityType', 1)) - 1
                        purpose_idx = max(0, min(9, purpose_idx))
                    except:
                        purpose_idx = 0
                    activities[idx, 2 + purpose_idx] = 1
                    
                    # 方式 one-hot
                    try:
                        mode_idx = int(act.get('ModelMode', 1)) - 1
                        mode_idx = max(0, min(10, mode_idx))
                    except:
                        mode_idx = 0
                    activities[idx, 12 + mode_idx] = 1
                    
                    # 是否驾驶员
                    try:
                        is_driver = int(act.get('是驾驶员还是乘客', 0))
                        is_driver = min(1, max(0, is_driver))
                    except:
                        is_driver = 0
                    activities[idx, 23 + is_driver] = 1
                    
                    # 是否联合出行
                    try:
                        is_joint = int(act.get('是否和家庭成员的联合出行', 0))
                        is_joint = min(1, max(0, is_joint))
                    except:
                        is_joint = 0
                    activities[idx, 25 + is_joint] = 1
                    
                    mask[idx] = 1
                
                member_list.append({
                    'person_id': f"{hh_id}_{member_id}",
                    'activities': activities,
                    'mask': mask
                })
            
            # 家庭属性
            try:
                n_workers = int(hh_info.get('家庭工作人口数', 0))
            except:
                n_workers = 0
            
            try:
                n_vehicles = int(hh_info.get('机动车数量', 0))
            except:
                n_vehicles = 0
            
            household_data[hh_id] = {
                'info': {
                    'n_members': len(members),
                    'n_workers': n_workers,
                    'n_vehicles': n_vehicles
                },
                'members': member_list
            }
        
        return household_data, household_ids
    
    def _extract_all_features(
        self,
        household_data: Dict[str, Dict],
        show_progress: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """从所有家庭提取特征"""
        
        person_features = []
        household_features = []
        
        iterator = tqdm(
            household_data.items(), 
            desc="提取特征", 
            disable=not show_progress,
            total=len(household_data)
        )
        
        for hh_id, hh_data in iterator:
            # 提取家庭成员特征
            member_activities = []
            member_masks = []
            
            for member in hh_data['members']:
                # 个人特征
                pf = self.person_extractor.extract_features(
                    person_id=member['person_id'],
                    household_id=hh_id,
                    activities=member['activities'],
                    mask=member['mask']
                )
                person_features.append(pf)
                
                member_activities.append(member['activities'])
                member_masks.append(member['mask'])
            
            # 家庭特征
            hf = self.household_extractor.extract_features(
                household_id=hh_id,
                household_info=hh_data['info'],
                member_activities=member_activities,
                member_masks=member_masks
            )
            household_features.append(hf)
        
        return person_features, household_features
    
    def fit_from_raw_data(
        self,
        family_df: pd.DataFrame,
        member_df: pd.DataFrame,
        activity_df: pd.DataFrame,
        auto_select_k: bool = True,
        person_n_components: Optional[int] = None,
        household_n_components: Optional[int] = None,
        show_progress: bool = True
    ) -> 'ActivityPatternExtractor':
        """
        从原始数据训练模式识别模型
        
        参数:
            family_df: 家庭信息DataFrame
            member_df: 成员信息DataFrame
            activity_df: 活动信息DataFrame
            auto_select_k: 是否自动选择聚类数
            person_n_components: 个人模式聚类数（auto_select_k=False时使用）
            household_n_components: 家庭模式聚类数（auto_select_k=False时使用）
            show_progress: 是否显示进度条
        """
        
        print("=" * 50)
        print("开始训练活动模式识别模型")
        print("=" * 50)
        
        # 步骤1: 准备数据
        print("\n[1/4] 准备数据结构...")
        household_data, self.household_ids = self._prepare_data_from_raw(
            family_df, member_df, activity_df,
            show_progress=show_progress
        )
        
        # 步骤2: 提取特征
        print("\n[2/4] 提取特征...")
        self.person_features_list, self.household_features_list = self._extract_all_features(
            household_data,
            show_progress=show_progress
        )
        
        # 步骤3: 训练个人模式GMM
        print("\n[3/4] 训练个人活动模式GMM...")
        person_X = np.vstack(self.person_features_list)
        self.person_gmm.fit(
            person_X,
            auto_select_k=auto_select_k,
            n_components=person_n_components,
            show_progress=show_progress
        )
        
        # 步骤4: 训练家庭模式GMM
        print("\n[4/4] 训练家庭活动模式GMM...")
        household_X = np.vstack(self.household_features_list)
        self.household_gmm.fit(
            household_X,
            auto_select_k=auto_select_k,
            n_components=household_n_components,
            show_progress=show_progress
        )
        
        print("\n" + "=" * 50)
        print("模型训练完成！")
        print(f"  个人模式数: {self.person_gmm.best_n_components}")
        print(f"  家庭模式数: {self.household_gmm.best_n_components}")
        print("=" * 50)
        
        return self
    
    def extract_all_patterns_from_raw(
        self,
        family_df: pd.DataFrame,
        member_df: pd.DataFrame,
        activity_df: pd.DataFrame,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        从原始数据提取所有模式
        
        返回:
            family_patterns: [n_households, n_household_patterns] 家庭模式概率
            individual_patterns: [n_households, max_members, n_person_patterns] 个人模式概率
            household_mapping: {household_id: batch_index} 映射关系
        """
        
        print("开始提取活动模式...")
        
        # 准备数据
        household_data, household_ids = self._prepare_data_from_raw(
            family_df, member_df, activity_df,
            show_progress=show_progress
        )
        
        n_households = len(household_ids)
        n_person_patterns = self.person_gmm.best_n_components
        n_household_patterns = self.household_gmm.best_n_components
        max_members = self.config.max_family_size
        
        # 初始化输出
        family_patterns = np.zeros((n_households, n_household_patterns))
        individual_patterns = np.zeros((n_households, max_members, n_person_patterns))
        household_mapping = {}
        
        iterator = tqdm(
            enumerate(household_ids),
            desc="提取模式",
            disable=not show_progress,
            total=n_households
        )
        
        for batch_idx, hh_id in iterator:
            household_mapping[hh_id] = batch_idx
            hh_data = household_data[hh_id]
            
            # 提取并预测家庭模式
            member_activities = [m['activities'] for m in hh_data['members']]
            member_masks = [m['mask'] for m in hh_data['members']]
            
            hf = self.household_extractor.extract_features(
                hh_id, hh_data['info'], member_activities, member_masks
            )
            family_patterns[batch_idx] = self.household_gmm.predict_proba(hf.reshape(1, -1))[0]
            
            # 提取并预测个人模式
            for member_idx, member in enumerate(hh_data['members']):
                if member_idx >= max_members:
                    break
                
                pf = self.person_extractor.extract_features(
                    member['person_id'], hh_id,
                    member['activities'], member['mask']
                )
                individual_patterns[batch_idx, member_idx] = self.person_gmm.predict_proba(
                    pf.reshape(1, -1)
                )[0]
        
        print(f"模式提取完成: {n_households} 个家庭")
        
        return family_patterns, individual_patterns, household_mapping
    
    def get_pattern_interpretations(self) -> Tuple[Dict[int, str], Dict[int, str]]:
        """获取模式解释"""
        
        # 个人模式解释
        person_centers = self.person_gmm.get_cluster_centers()
        person_interp = {}
        
        for i, center in enumerate(person_centers):
            n_activities = center[0]
            purpose_dist = center[6:16]
            main_purpose = np.argmax(purpose_dist)
            joint_ratio = center[-1]
            
            purpose_names = ['home', 'work', 'school', 'business', 'meal', 
                           'social', 'leisure', 'shopping', 'escort', 'other']
            
            if n_activities < 1.5:
                pattern = "居家型"
            elif main_purpose == 1:
                pattern = "通勤型"
            elif main_purpose == 2:
                pattern = "学生型"
            elif main_purpose in [6, 7]:
                pattern = "休闲型"
            else:
                pattern = "混合型"
            
            if joint_ratio > 0.3:
                pattern += "(联合出行)"
            
            person_interp[i] = pattern
        
        # 家庭模式解释
        household_centers = self.household_gmm.get_cluster_centers()
        household_interp = {}
        
        for i, center in enumerate(household_centers):
            n_members = center[0]
            n_workers = center[1]
            car_usage = center[3]
            joint_ratio = center[4]
            
            if n_members <= 2:
                size = "小家庭"
            elif n_members <= 4:
                size = "中等家庭"
            else:
                size = "大家庭"
            
            if car_usage > 0.5:
                mobility = "高车辆依赖"
            elif car_usage > 0.2:
                mobility = "混合出行"
            else:
                mobility = "低车辆依赖"
            
            if joint_ratio > 0.3:
                interaction = "高互动"
            else:
                interaction = "独立出行"
            
            household_interp[i] = f"{size}, {mobility}, {interaction}"
        
        return person_interp, household_interp
    
    def get_pattern_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取模式embedding（用于decoder）"""
        person_emb = self.person_gmm.get_cluster_centers()
        household_emb = self.household_gmm.get_cluster_centers()
        return person_emb, household_emb
    
    def save_model(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'person_gmm': self.person_gmm,
                'household_gmm': self.household_gmm,
                'household_ids': self.household_ids
            }, f)
        print(f"模型已保存至: {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'ActivityPatternExtractor':
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls(data['config'])
        extractor.person_gmm = data['person_gmm']
        extractor.household_gmm = data['household_gmm']
        extractor.household_ids = data['household_ids']
        
        print(f"模型已从 {path} 加载")
        return extractor
    
    # 保持与原代码的兼容性
    @property
    def pattern_pipeline(self):
        """兼容性属性，模拟原来的pipeline结构"""
        class _CompatPipeline:
            def __init__(self, extractor):
                self.person_gmm = extractor.person_gmm
                self.household_gmm = extractor.household_gmm
        return _CompatPipeline(self)


# ==================== 辅助函数 ====================

def load_data_from_notebook():
    """
    从notebook环境加载数据的辅助函数
    这个函数需要根据实际notebook中的变量名调整
    """
    import sys
    
    # 尝试从IPython环境获取变量
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            user_ns = ipython.user_ns
            return (
                user_ns.get('family2023'),
                user_ns.get('familymember_2023'),
                user_ns.get('activityinfo')
            )
    except:
        pass
    
    return None, None, None


# ==================== 测试 ====================

if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    
    n_families = 50
    
    # 模拟家庭数据
    family_data = {
        '家庭编号': [f'HH_{i:04d}' for i in range(n_families)],
        '家庭工作人口数': np.random.randint(0, 3, n_families).astype(str),
        '机动车数量': np.random.randint(0, 3, n_families).astype(str)
    }
    family_df = pd.DataFrame(family_data)
    
    # 模拟成员数据
    member_records = []
    for i in range(n_families):
        n_members = np.random.randint(1, 5)
        for j in range(n_members):
            member_records.append({
                '家庭编号': f'HH_{i:04d}',
                '家庭成员编号': f'{j}'
            })
    member_df = pd.DataFrame(member_records)
    
    # 模拟活动数据
    activity_records = []
    for i in range(n_families):
        members = member_df[member_df['家庭编号'] == f'HH_{i:04d}']
        for _, member in members.iterrows():
            n_acts = np.random.randint(1, 5)
            for k in range(n_acts):
                activity_records.append({
                    '家庭编号': f'HH_{i:04d}',
                    '家庭成员编号': member['家庭成员编号'],
                    '出行序号': str(k),
                    '出发时间1小时时间段': str(np.random.randint(6, 22)),
                    '到达时间1小时时间段': str(np.random.randint(7, 23)),
                    'ActivityType': str(np.random.randint(1, 11)),
                    'ModelMode': str(np.random.randint(1, 12)),
                    '是驾驶员还是乘客': str(np.random.randint(0, 2)),
                    '是否和家庭成员的联合出行': str(np.random.randint(0, 2))
                })
    activity_df = pd.DataFrame(activity_records)
    
    print(f"测试数据: {len(family_df)} 家庭, {len(member_df)} 成员, {len(activity_df)} 活动")
    
    # 测试训练
    config = PatternExtractionConfig(
        person_n_components_range=(2, 8),
        household_n_components_range=(2, 6)
    )
    
    extractor = ActivityPatternExtractor(config)
    extractor.fit_from_raw_data(
        family_df, member_df, activity_df,
        auto_select_k=True,
        show_progress=True
    )
    
    # 测试模式提取
    family_patterns, individual_patterns, mapping = extractor.extract_all_patterns_from_raw(
        family_df, member_df, activity_df,
        show_progress=True
    )
    
    print(f"\n结果:")
    print(f"  家庭模式矩阵: {family_patterns.shape}")
    print(f"  个人模式矩阵: {individual_patterns.shape}")
    
    # 模式解释
    person_interp, household_interp = extractor.get_pattern_interpretations()
    print("\n个人模式解释:")
    for i, desc in person_interp.items():
        print(f"  模式 {i}: {desc}")
    
    print("\n家庭模式解释:")
    for i, desc in household_interp.items():
        print(f"  模式 {i}: {desc}")
