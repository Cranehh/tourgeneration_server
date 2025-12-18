"""
家庭和个人活动模式GMM聚类模型

该模块实现两个层次的GMM聚类：
1. 个人活动模式GMM：基于个人日活动链特征
2. 家庭活动模式GMM：基于家庭层面的车辆使用、联合出行等特征

作者：郝赫
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ==================== 数据结构定义 ====================

@dataclass
class ActivityFeatureConfig:
    """活动特征配置"""
    # 连续变量维度
    continuous_dims: int = 2  # 开始时间z-score, 结束时间z-score
    
    # 离散变量类别数
    purpose_classes: int = 10    # 活动目的类别
    mode_classes: int = 11       # 出行方式类别
    driver_classes: int = 2      # 是否驾驶员
    joint_classes: int = 2       # 是否联合出行
    
    @property
    def total_dims(self) -> int:
        """总特征维度 = 2 + 10 + 11 + 2 + 2 = 27"""
        return (self.continuous_dims + self.purpose_classes + 
                self.mode_classes + self.driver_classes + self.joint_classes)


@dataclass
class PersonActivityFeatures:
    """个人活动特征（从原始活动序列提取的统计特征）"""
    person_id: str
    household_id: str
    
    # ====== 活动链结构特征 ======
    n_activities: int                    # 活动数量
    n_tours: int                         # tour数量（从家出发返回家的次数）
    tour_complexity: float               # tour复杂度（平均每个tour的停靠点数）
    
    # ====== 时间特征 ======
    first_departure_time: float          # 首次出发时间（z-score）
    last_return_time: float              # 最后返家时间（z-score）
    total_out_of_home_duration: float    # 总外出时长（z-score）
    avg_activity_duration: float         # 平均活动持续时长（z-score）
    time_entropy: float                  # 时间分布熵（活动在一天中的分散程度）
    
    # ====== 活动目的分布 ======
    purpose_distribution: np.ndarray     # 各目的占比 [10维]
    primary_purpose: int                 # 主要活动目的（出现最多的）
    purpose_diversity: float             # 目的多样性（熵）
    
    # ====== 出行方式分布 ======
    mode_distribution: np.ndarray        # 各方式占比 [11维]
    primary_mode: int                    # 主要出行方式
    mode_diversity: float                # 方式多样性
    
    # ====== 联合出行特征 ======
    joint_trip_ratio: float              # 联合出行比例
    n_joint_activities: int              # 联合活动数量
    
    # ====== 驾驶特征 ======
    driver_ratio: float                  # 作为驾驶员的比例
    
    def to_vector(self) -> np.ndarray:
        """转换为GMM输入向量"""
        return np.concatenate([
            # 活动链结构 [3维]
            np.array([self.n_activities, self.n_tours, self.tour_complexity]),
            
            # 时间特征 [5维]
            np.array([
                self.first_departure_time,
                self.last_return_time, 
                self.total_out_of_home_duration,
                self.avg_activity_duration,
                self.time_entropy
            ]),
            
            # 活动目的分布 [10维] + 多样性 [1维]
            self.purpose_distribution,
            np.array([self.purpose_diversity]),
            
            # 出行方式分布 [11维] + 多样性 [1维]
            self.mode_distribution,
            np.array([self.mode_diversity]),
            
            # 联合出行和驾驶特征 [3维]
            np.array([self.joint_trip_ratio, self.n_joint_activities, self.driver_ratio])
        ])
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """获取特征名称列表"""
        names = [
            # 活动链结构
            'n_activities', 'n_tours', 'tour_complexity',
            # 时间特征
            'first_departure_time', 'last_return_time', 
            'total_out_of_home_duration', 'avg_activity_duration', 'time_entropy',
        ]
        # 目的分布
        purpose_names = ['purpose_home', 'purpose_work', 'purpose_school', 
                        'purpose_shopping', 'purpose_leisure', 'purpose_personal',
                        'purpose_social', 'purpose_escort', 'purpose_meal', 'purpose_other']
        names.extend(purpose_names)
        names.append('purpose_diversity')
        
        # 方式分布
        mode_names = ['mode_walk', 'mode_bike', 'mode_car_driver', 'mode_car_passenger',
                     'mode_bus', 'mode_metro', 'mode_taxi', 'mode_rideshare',
                     'mode_motorcycle', 'mode_other', 'mode_multimodal']
        names.extend(mode_names)
        names.append('mode_diversity')
        
        # 联合出行和驾驶
        names.extend(['joint_trip_ratio', 'n_joint_activities', 'driver_ratio'])
        
        return names
    
    @staticmethod
    def get_feature_dim() -> int:
        """获取特征总维度: 3 + 5 + 11 + 12 + 3 = 34"""
        return 34


@dataclass
class HouseholdActivityFeatures:
    """家庭活动特征"""
    household_id: str
    
    # ====== 家庭基本信息 ======
    n_members: int                       # 家庭成员数
    n_workers: int                       # 工作者数量
    n_students: int                      # 学生数量
    n_vehicles: int                      # 车辆数量
    
    # ====== 车辆使用特征 ======
    vehicle_usage_rate: float            # 车辆使用率（使用天数/总天数）
    avg_daily_vehicle_trips: float       # 日均车辆出行次数
    vehicle_time_distribution: np.ndarray  # 车辆使用时段分布 [4维: 早/中/晚/夜]
    vehicle_purpose_distribution: np.ndarray  # 车辆使用目的分布 [简化为5维]
    
    # ====== 联合出行特征 ======
    joint_trip_frequency: float          # 联合出行频率（次/天）
    avg_joint_participants: float        # 平均联合出行参与人数
    joint_trip_purpose_dist: np.ndarray  # 联合出行目的分布 [5维]
    intra_household_interaction: float   # 家庭内部互动强度
    
    # ====== 家庭活动协调特征 ======
    activity_sync_score: float           # 活动同步得分（成员间时间重叠度）
    escort_trip_ratio: float             # 接送出行比例
    shared_destination_ratio: float      # 共享目的地比例
    
    # ====== 家庭时间利用特征 ======
    household_first_departure: float     # 家庭最早出发时间
    household_last_return: float         # 家庭最晚返家时间
    household_active_window: float       # 家庭活动时间窗口长度
    
    def to_vector(self) -> np.ndarray:
        """转换为GMM输入向量"""
        return np.concatenate([
            # 家庭结构 [4维]
            np.array([self.n_members, self.n_workers, self.n_students, self.n_vehicles]),
            
            # 车辆使用 [2 + 4 + 5 = 11维]
            np.array([self.vehicle_usage_rate, self.avg_daily_vehicle_trips]),
            self.vehicle_time_distribution,
            self.vehicle_purpose_distribution,
            
            # 联合出行 [3 + 5 + 1 = 9维]
            np.array([self.joint_trip_frequency, self.avg_joint_participants, 
                     self.intra_household_interaction]),
            self.joint_trip_purpose_dist,
            
            # 活动协调 [3维]
            np.array([self.activity_sync_score, self.escort_trip_ratio, 
                     self.shared_destination_ratio]),
            
            # 时间利用 [3维]
            np.array([self.household_first_departure, self.household_last_return,
                     self.household_active_window])
        ])
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """获取特征名称列表"""
        names = [
            # 家庭结构
            'n_members', 'n_workers', 'n_students', 'n_vehicles',
            # 车辆使用
            'vehicle_usage_rate', 'avg_daily_vehicle_trips',
            'vehicle_time_morning', 'vehicle_time_midday', 
            'vehicle_time_evening', 'vehicle_time_night',
            'vehicle_purpose_work', 'vehicle_purpose_school',
            'vehicle_purpose_shopping', 'vehicle_purpose_leisure', 'vehicle_purpose_other',
            # 联合出行
            'joint_trip_frequency', 'avg_joint_participants', 'intra_household_interaction',
            'joint_purpose_work', 'joint_purpose_school', 
            'joint_purpose_shopping', 'joint_purpose_leisure', 'joint_purpose_other',
            # 活动协调
            'activity_sync_score', 'escort_trip_ratio', 'shared_destination_ratio',
            # 时间利用
            'household_first_departure', 'household_last_return', 'household_active_window'
        ]
        return names
    
    @staticmethod
    def get_feature_dim() -> int:
        """获取特征总维度: 4 + 11 + 8 + 3 + 3 = 29"""
        # 注：实际计算 4 + (2+4+5) + (3+5) + 3 + 3 = 4 + 11 + 8 + 3 + 3 = 29
        return 29


# ==================== 特征提取器 ====================

class PersonFeatureExtractor:
    """从原始活动序列提取个人特征"""
    
    def __init__(self, config: ActivityFeatureConfig = None):
        self.config = config or ActivityFeatureConfig()
        
    def extract_from_activity_sequence(
        self, 
        person_id: str,
        household_id: str,
        activities: np.ndarray,  # [n_activities, 27] 原始活动特征
        activity_mask: np.ndarray  # [n_activities] 有效活动mask
    ) -> PersonActivityFeatures:
        """
        从原始活动序列提取统计特征
        
        activities格式：[n_activities, 27]
            - [:, 0:2]: 开始时间, 结束时间 (z-score)
            - [:, 2:12]: 目的one-hot (10类)
            - [:, 12:23]: 方式one-hot (11类)
            - [:, 23:25]: 是否驾驶员one-hot (2类)
            - [:, 25:27]: 是否联合出行one-hot (2类)
        """
        # 获取有效活动
        valid_activities = activities[activity_mask.astype(bool)]
        n_activities = len(valid_activities)
        
        if n_activities == 0:
            return self._get_empty_features(person_id, household_id)
        
        # 提取各维度
        start_times = valid_activities[:, 0]
        end_times = valid_activities[:, 1]
        purposes = valid_activities[:, 2:12]  # one-hot
        modes = valid_activities[:, 12:23]     # one-hot
        is_driver = valid_activities[:, 23:25]  # one-hot
        is_joint = valid_activities[:, 25:27]   # one-hot
        
        # 计算tour数量（简化：假设目的为home表示tour结束）
        purpose_indices = np.argmax(purposes, axis=1)
        home_returns = np.sum(purpose_indices == 0)  # 假设0是home
        n_tours = max(1, home_returns)
        
        # 时间特征
        first_departure = start_times[0] if len(start_times) > 0 else 0
        last_return = end_times[-1] if len(end_times) > 0 else 0
        durations = end_times - start_times
        total_duration = np.sum(np.maximum(durations, 0))
        avg_duration = np.mean(np.maximum(durations, 0)) if n_activities > 0 else 0
        
        # 时间熵（活动在一天中的分散程度）
        time_bins = np.histogram(start_times, bins=24, range=(-3, 3))[0]
        time_probs = time_bins / (np.sum(time_bins) + 1e-10)
        time_entropy = -np.sum(time_probs * np.log(time_probs + 1e-10))
        
        # 目的分布和多样性
        purpose_dist = np.mean(purposes, axis=0)
        purpose_dist = purpose_dist / (np.sum(purpose_dist) + 1e-10)
        purpose_diversity = -np.sum(purpose_dist * np.log(purpose_dist + 1e-10))
        primary_purpose = np.argmax(purpose_dist)
        
        # 方式分布和多样性
        mode_dist = np.mean(modes, axis=0)
        mode_dist = mode_dist / (np.sum(mode_dist) + 1e-10)
        mode_diversity = -np.sum(mode_dist * np.log(mode_dist + 1e-10))
        primary_mode = np.argmax(mode_dist)
        
        # 联合出行特征
        joint_flags = np.argmax(is_joint, axis=1)  # 1表示联合出行
        joint_ratio = np.mean(joint_flags)
        n_joint = np.sum(joint_flags)
        
        # 驾驶特征
        driver_flags = np.argmax(is_driver, axis=1)  # 1表示是驾驶员
        driver_ratio = np.mean(driver_flags)
        
        # tour复杂度
        tour_complexity = n_activities / n_tours if n_tours > 0 else 0
        
        return PersonActivityFeatures(
            person_id=person_id,
            household_id=household_id,
            n_activities=n_activities,
            n_tours=n_tours,
            tour_complexity=tour_complexity,
            first_departure_time=first_departure,
            last_return_time=last_return,
            total_out_of_home_duration=total_duration,
            avg_activity_duration=avg_duration,
            time_entropy=time_entropy,
            purpose_distribution=purpose_dist,
            primary_purpose=primary_purpose,
            purpose_diversity=purpose_diversity,
            mode_distribution=mode_dist,
            primary_mode=primary_mode,
            mode_diversity=mode_diversity,
            joint_trip_ratio=joint_ratio,
            n_joint_activities=n_joint,
            driver_ratio=driver_ratio
        )
    
    def _get_empty_features(self, person_id: str, household_id: str) -> PersonActivityFeatures:
        """返回空特征（无活动的情况）"""
        return PersonActivityFeatures(
            person_id=person_id,
            household_id=household_id,
            n_activities=0,
            n_tours=0,
            tour_complexity=0,
            first_departure_time=0,
            last_return_time=0,
            total_out_of_home_duration=0,
            avg_activity_duration=0,
            time_entropy=0,
            purpose_distribution=np.zeros(10),
            primary_purpose=0,
            purpose_diversity=0,
            mode_distribution=np.zeros(11),
            primary_mode=0,
            mode_diversity=0,
            joint_trip_ratio=0,
            n_joint_activities=0,
            driver_ratio=0
        )


class HouseholdFeatureExtractor:
    """从家庭成员活动序列提取家庭特征"""
    
    def __init__(self, config: ActivityFeatureConfig = None):
        self.config = config or ActivityFeatureConfig()
        
    def extract_from_household_activities(
        self,
        household_id: str,
        household_info: Dict,  # 家庭基本信息
        member_activities: List[np.ndarray],  # 各成员活动序列
        member_masks: List[np.ndarray]  # 各成员活动mask
    ) -> HouseholdActivityFeatures:
        """
        从家庭所有成员的活动序列提取家庭层面特征
        
        household_info包含:
            - n_members: 成员数
            - n_workers: 工作者数
            - n_students: 学生数
            - n_vehicles: 车辆数
        """
        n_members = household_info.get('n_members', len(member_activities))
        n_workers = household_info.get('n_workers', 0)
        n_students = household_info.get('n_students', 0)
        n_vehicles = household_info.get('n_vehicles', 0)
        
        # 收集所有有效活动
        all_activities = []
        all_start_times = []
        all_end_times = []
        all_purposes = []
        all_modes = []
        all_is_driver = []
        all_is_joint = []
        
        for activities, mask in zip(member_activities, member_masks):
            valid = activities[mask.astype(bool)]
            if len(valid) > 0:
                all_activities.append(valid)
                all_start_times.extend(valid[:, 0])
                all_end_times.extend(valid[:, 1])
                all_purposes.append(valid[:, 2:12])
                all_modes.append(valid[:, 12:23])
                all_is_driver.append(valid[:, 23:25])
                all_is_joint.append(valid[:, 25:27])
        
        if len(all_activities) == 0:
            return self._get_empty_features(household_id, household_info)
        
        # 合并所有活动
        all_purposes = np.vstack(all_purposes)
        all_modes = np.vstack(all_modes)
        all_is_driver = np.vstack(all_is_driver)
        all_is_joint = np.vstack(all_is_joint)
        all_start_times = np.array(all_start_times)
        all_end_times = np.array(all_end_times)
        
        # ====== 车辆使用特征 ======
        # 识别使用车辆的活动（mode为car_driver或car_passenger）
        mode_indices = np.argmax(all_modes, axis=1)
        car_activities = (mode_indices == 2) | (mode_indices == 3)  # 假设2是car_driver, 3是car_passenger
        
        vehicle_usage_rate = np.mean(car_activities) if len(car_activities) > 0 else 0
        avg_daily_vehicle_trips = np.sum(car_activities)  # 简化处理
        
        # 车辆使用时段分布 [早6-9, 中9-17, 晚17-21, 夜21-6]
        car_start_times = all_start_times[car_activities] if np.any(car_activities) else np.array([])
        vehicle_time_dist = self._compute_time_distribution(car_start_times)
        
        # 车辆使用目的分布（简化为5类）
        if np.any(car_activities):
            car_purposes = all_purposes[car_activities]
            vehicle_purpose_dist = self._simplify_purpose_distribution(car_purposes)
        else:
            vehicle_purpose_dist = np.zeros(5)
        
        # ====== 联合出行特征 ======
        joint_flags = np.argmax(all_is_joint, axis=1)
        joint_activities = all_is_joint[joint_flags == 1] if np.any(joint_flags) else np.array([])
        
        joint_trip_frequency = np.sum(joint_flags) / max(n_members, 1)
        avg_joint_participants = self._estimate_joint_participants(
            member_activities, member_masks
        )
        
        # 联合出行目的分布
        if np.any(joint_flags):
            joint_purposes = all_purposes[joint_flags == 1]
            joint_purpose_dist = self._simplify_purpose_distribution(joint_purposes)
        else:
            joint_purpose_dist = np.zeros(5)
        
        # 家庭内部互动强度（联合活动占比）
        intra_interaction = np.mean(joint_flags) if len(joint_flags) > 0 else 0
        
        # ====== 活动协调特征 ======
        activity_sync = self._compute_activity_sync(member_activities, member_masks)
        
        # 接送出行比例（假设purpose=7是escort）
        purpose_indices = np.argmax(all_purposes, axis=1)
        escort_ratio = np.mean(purpose_indices == 7)
        
        # 共享目的地比例（简化计算）
        shared_dest_ratio = self._compute_shared_destination_ratio(
            member_activities, member_masks
        )
        
        # ====== 家庭时间利用特征 ======
        household_first_dep = np.min(all_start_times) if len(all_start_times) > 0 else 0
        household_last_ret = np.max(all_end_times) if len(all_end_times) > 0 else 0
        household_active_window = household_last_ret - household_first_dep
        
        return HouseholdActivityFeatures(
            household_id=household_id,
            n_members=n_members,
            n_workers=n_workers,
            n_students=n_students,
            n_vehicles=n_vehicles,
            vehicle_usage_rate=vehicle_usage_rate,
            avg_daily_vehicle_trips=avg_daily_vehicle_trips,
            vehicle_time_distribution=vehicle_time_dist,
            vehicle_purpose_distribution=vehicle_purpose_dist,
            joint_trip_frequency=joint_trip_frequency,
            avg_joint_participants=avg_joint_participants,
            joint_trip_purpose_dist=joint_purpose_dist,
            intra_household_interaction=intra_interaction,
            activity_sync_score=activity_sync,
            escort_trip_ratio=escort_ratio,
            shared_destination_ratio=shared_dest_ratio,
            household_first_departure=household_first_dep,
            household_last_return=household_last_ret,
            household_active_window=household_active_window
        )
    
    def _compute_time_distribution(self, start_times: np.ndarray) -> np.ndarray:
        """计算时段分布 [早6-9, 中9-17, 晚17-21, 夜21-6]"""
        if len(start_times) == 0:
            return np.zeros(4)
        
        # 假设z-score的0对应中午12点，标准差为6小时
        # 转换回实际时间：time_hour = start_time * 6 + 12
        hours = start_times * 6 + 12
        hours = np.clip(hours, 0, 24)
        
        dist = np.zeros(4)
        dist[0] = np.sum((hours >= 6) & (hours < 9))    # 早
        dist[1] = np.sum((hours >= 9) & (hours < 17))   # 中
        dist[2] = np.sum((hours >= 17) & (hours < 21))  # 晚
        dist[3] = np.sum((hours >= 21) | (hours < 6))   # 夜
        
        total = np.sum(dist)
        return dist / (total + 1e-10)
    
    def _simplify_purpose_distribution(self, purposes: np.ndarray) -> np.ndarray:
        """将10类目的简化为5类: work, school, shopping, leisure, other"""
        if len(purposes) == 0:
            return np.zeros(5)
        
        # 假设原始10类映射：
        # 0-home, 1-work, 2-school, 3-shopping, 4-leisure, 
        # 5-personal, 6-social, 7-escort, 8-meal, 9-other
        
        purpose_indices = np.argmax(purposes, axis=1)
        simplified = np.zeros(5)
        simplified[0] = np.sum(purpose_indices == 1)  # work
        simplified[1] = np.sum(purpose_indices == 2)  # school
        simplified[2] = np.sum(purpose_indices == 3)  # shopping
        simplified[3] = np.sum((purpose_indices == 4) | (purpose_indices == 6) | 
                               (purpose_indices == 8))  # leisure (包括social, meal)
        simplified[4] = np.sum((purpose_indices == 0) | (purpose_indices == 5) | 
                               (purpose_indices == 7) | (purpose_indices == 9))  # other
        
        total = np.sum(simplified)
        return simplified / (total + 1e-10)
    
    def _estimate_joint_participants(
        self, 
        member_activities: List[np.ndarray],
        member_masks: List[np.ndarray]
    ) -> float:
        """估计平均联合出行参与人数"""
        joint_counts = []
        for activities, mask in zip(member_activities, member_masks):
            valid = activities[mask.astype(bool)]
            if len(valid) > 0:
                joint_flags = np.argmax(valid[:, 25:27], axis=1)
                joint_counts.append(np.sum(joint_flags))
        
        if len(joint_counts) == 0:
            return 0
        
        total_joint = np.sum(joint_counts)
        if total_joint == 0:
            return 0
        
        # 简化估计：假设联合出行平均2人参与
        return 2.0
    
    def _compute_activity_sync(
        self,
        member_activities: List[np.ndarray],
        member_masks: List[np.ndarray]
    ) -> float:
        """计算成员间活动同步得分"""
        if len(member_activities) < 2:
            return 0
        
        # 收集各成员的活动时间段
        member_time_ranges = []
        for activities, mask in zip(member_activities, member_masks):
            valid = activities[mask.astype(bool)]
            if len(valid) > 0:
                starts = valid[:, 0]
                ends = valid[:, 1]
                member_time_ranges.append((starts, ends))
        
        if len(member_time_ranges) < 2:
            return 0
        
        # 计算时间重叠度（简化版本）
        overlap_scores = []
        for i in range(len(member_time_ranges)):
            for j in range(i + 1, len(member_time_ranges)):
                starts_i, ends_i = member_time_ranges[i]
                starts_j, ends_j = member_time_ranges[j]
                
                # 计算活动时间的相关性
                mean_start_i = np.mean(starts_i)
                mean_start_j = np.mean(starts_j)
                overlap = 1 - abs(mean_start_i - mean_start_j) / 4  # 归一化
                overlap_scores.append(max(0, overlap))
        
        return np.mean(overlap_scores) if overlap_scores else 0
    
    def _compute_shared_destination_ratio(
        self,
        member_activities: List[np.ndarray],
        member_masks: List[np.ndarray]
    ) -> float:
        """计算共享目的地比例（基于目的类型的重叠）"""
        if len(member_activities) < 2:
            return 0
        
        # 收集各成员的主要目的
        member_purposes = []
        for activities, mask in zip(member_activities, member_masks):
            valid = activities[mask.astype(bool)]
            if len(valid) > 0:
                purposes = valid[:, 2:12]
                purpose_indices = set(np.argmax(purposes, axis=1))
                member_purposes.append(purpose_indices)
        
        if len(member_purposes) < 2:
            return 0
        
        # 计算目的类型的交集比例
        shared = member_purposes[0]
        for purposes in member_purposes[1:]:
            shared = shared & purposes
        
        all_purposes = member_purposes[0]
        for purposes in member_purposes[1:]:
            all_purposes = all_purposes | purposes
        
        if len(all_purposes) == 0:
            return 0
        
        return len(shared) / len(all_purposes)
    
    def _get_empty_features(
        self, 
        household_id: str, 
        household_info: Dict
    ) -> HouseholdActivityFeatures:
        """返回空特征"""
        return HouseholdActivityFeatures(
            household_id=household_id,
            n_members=household_info.get('n_members', 0),
            n_workers=household_info.get('n_workers', 0),
            n_students=household_info.get('n_students', 0),
            n_vehicles=household_info.get('n_vehicles', 0),
            vehicle_usage_rate=0,
            avg_daily_vehicle_trips=0,
            vehicle_time_distribution=np.zeros(4),
            vehicle_purpose_distribution=np.zeros(5),
            joint_trip_frequency=0,
            avg_joint_participants=0,
            joint_trip_purpose_dist=np.zeros(5),
            intra_household_interaction=0,
            activity_sync_score=0,
            escort_trip_ratio=0,
            shared_destination_ratio=0,
            household_first_departure=0,
            household_last_return=0,
            household_active_window=0
        )


# ==================== GMM聚类模型 ====================

class ActivityPatternGMM:
    """活动模式GMM聚类基类"""
    
    def __init__(
        self,
        n_components_range: Tuple[int, int] = (3, 15),
        covariance_type: str = 'full',
        random_state: int = 42
    ):
        self.n_components_range = n_components_range
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.gmm: Optional[GaussianMixture] = None
        self.best_n_components: Optional[int] = None
        self.feature_names: List[str] = []
        
    def _select_n_components(
        self, 
        X: np.ndarray,
        method: str = 'bic'
    ) -> int:
        """通过BIC/AIC或轮廓系数选择最佳聚类数"""
        scores = []
        n_range = range(self.n_components_range[0], self.n_components_range[1] + 1)
        
        for n in n_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=3
            )
            gmm.fit(X)
            
            if method == 'bic':
                scores.append(gmm.bic(X))
            elif method == 'aic':
                scores.append(gmm.aic(X))
            elif method == 'silhouette':
                labels = gmm.predict(X)
                if len(np.unique(labels)) > 1:
                    scores.append(-silhouette_score(X, labels))  # 负号使其与BIC/AIC一致
                else:
                    scores.append(np.inf)
        
        best_idx = np.argmin(scores)
        return list(n_range)[best_idx]
    
    def fit(
        self, 
        features: np.ndarray,
        auto_select_k: bool = True,
        n_components: Optional[int] = None
    ) -> 'ActivityPatternGMM':
        """训练GMM模型"""
        # 标准化
        X = self.scaler.fit_transform(features)
        
        # 选择聚类数
        if auto_select_k:
            self.best_n_components = self._select_n_components(X)
            print(f"Auto-selected n_components: {self.best_n_components}")
        else:
            self.best_n_components = n_components or self.n_components_range[0]
        
        # 训练最终模型
        self.gmm = GaussianMixture(
            n_components=self.best_n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5
        )
        self.gmm.fit(X)
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测聚类标签"""
        X = self.scaler.transform(features)
        return self.gmm.predict(X)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测聚类概率（软标签）"""
        X = self.scaler.transform(features)
        return self.gmm.predict_proba(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        """获取聚类中心（原始特征空间）"""
        return self.scaler.inverse_transform(self.gmm.means_)
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """获取聚类摘要"""
        centers = self.get_cluster_centers()
        summary = pd.DataFrame(
            centers,
            columns=self.feature_names,
            index=[f'Pattern_{i}' for i in range(len(centers))]
        )
        # 添加聚类权重
        summary['weight'] = self.gmm.weights_
        return summary


class PersonPatternGMM(ActivityPatternGMM):
    """个人活动模式GMM"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_names = PersonActivityFeatures.get_feature_names()
        
    def fit_from_features(
        self,
        person_features: List[PersonActivityFeatures],
        **kwargs
    ) -> 'PersonPatternGMM':
        """从PersonActivityFeatures列表训练"""
        X = np.vstack([f.to_vector() for f in person_features])
        return self.fit(X, **kwargs)
    
    def interpret_patterns(self) -> Dict[int, str]:
        """解释各模式的含义"""
        centers = self.get_cluster_centers()
        interpretations = {}
        
        for i, center in enumerate(centers):
            # 解析关键特征
            n_activities = center[0]
            n_tours = center[1]
            first_dep = center[3]
            
            # 找到主要目的（目的分布从index 8开始，共10维）
            purpose_start = 8
            purpose_dist = center[purpose_start:purpose_start + 10]
            main_purpose_idx = np.argmax(purpose_dist)
            purpose_names = ['home', 'work', 'school', 'shopping', 'leisure',
                           'personal', 'social', 'escort', 'meal', 'other']
            main_purpose = purpose_names[main_purpose_idx]
            
            # 联合出行特征
            joint_ratio = center[-3]
            
            # 构建描述
            if n_activities < 2:
                pattern_type = "Stay-at-home"
            elif main_purpose == 'work':
                if n_tours > 1.5:
                    pattern_type = "Multi-tour Worker"
                else:
                    pattern_type = "Simple Commuter"
            elif main_purpose == 'school':
                pattern_type = "Student"
            elif main_purpose in ['shopping', 'leisure', 'social']:
                pattern_type = "Discretionary"
            else:
                pattern_type = "Mixed"
            
            if joint_ratio > 0.3:
                pattern_type += " (Joint-oriented)"
            
            interpretations[i] = pattern_type
        
        return interpretations


class HouseholdPatternGMM(ActivityPatternGMM):
    """家庭活动模式GMM"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_names = HouseholdActivityFeatures.get_feature_names()
        
    def fit_from_features(
        self,
        household_features: List[HouseholdActivityFeatures],
        **kwargs
    ) -> 'HouseholdPatternGMM':
        """从HouseholdActivityFeatures列表训练"""
        X = np.vstack([f.to_vector() for f in household_features])
        return self.fit(X, **kwargs)
    
    def interpret_patterns(self) -> Dict[int, str]:
        """解释各模式的含义"""
        centers = self.get_cluster_centers()
        interpretations = {}
        
        for i, center in enumerate(centers):
            # 解析关键特征
            n_members = center[0]
            n_workers = center[1]
            n_vehicles = center[3]
            vehicle_usage = center[4]
            joint_freq = center[15]  # joint_trip_frequency
            
            # 构建描述
            if n_members <= 1.5:
                size_type = "Single"
            elif n_members <= 2.5:
                size_type = "Couple"
            else:
                size_type = "Family"
            
            if vehicle_usage > 0.5:
                mobility_type = "Car-dependent"
            elif vehicle_usage > 0.2:
                mobility_type = "Mixed-mode"
            else:
                mobility_type = "Non-car"
            
            if joint_freq > 0.3:
                interaction_type = "High-interaction"
            else:
                interaction_type = "Independent"
            
            interpretations[i] = f"{size_type}, {mobility_type}, {interaction_type}"
        
        return interpretations


# ==================== 完整Pipeline ====================

class ActivityPatternPipeline:
    """活动模式识别完整Pipeline"""
    
    def __init__(
        self,
        person_gmm_config: Dict = None,
        household_gmm_config: Dict = None
    ):
        self.config = ActivityFeatureConfig()
        self.person_extractor = PersonFeatureExtractor(self.config)
        self.household_extractor = HouseholdFeatureExtractor(self.config)
        
        person_config = person_gmm_config or {}
        household_config = household_gmm_config or {}
        
        self.person_gmm = PersonPatternGMM(**person_config)
        self.household_gmm = HouseholdPatternGMM(**household_config)
        
        self.person_features: List[PersonActivityFeatures] = []
        self.household_features: List[HouseholdActivityFeatures] = []
        
    def extract_features(
        self,
        household_data: Dict[str, Dict]
    ) -> Tuple[List[PersonActivityFeatures], List[HouseholdActivityFeatures]]:
        """
        从原始数据提取特征
        
        household_data格式:
        {
            'household_id': {
                'info': {'n_members': int, 'n_workers': int, ...},
                'members': [
                    {
                        'person_id': str,
                        'activities': np.ndarray,  # [n_activities, 27]
                        'mask': np.ndarray  # [n_activities]
                    },
                    ...
                ]
            }
        }
        """
        person_features = []
        household_features = []
        
        for hh_id, hh_data in household_data.items():
            # 提取家庭成员特征
            member_activities = []
            member_masks = []
            
            for member in hh_data['members']:
                # 个人特征
                pf = self.person_extractor.extract_from_activity_sequence(
                    person_id=member['person_id'],
                    household_id=hh_id,
                    activities=member['activities'],
                    activity_mask=member['mask']
                )
                person_features.append(pf)
                
                member_activities.append(member['activities'])
                member_masks.append(member['mask'])
            
            # 家庭特征
            hf = self.household_extractor.extract_from_household_activities(
                household_id=hh_id,
                household_info=hh_data['info'],
                member_activities=member_activities,
                member_masks=member_masks
            )
            household_features.append(hf)
        
        self.person_features = person_features
        self.household_features = household_features
        
        return person_features, household_features
    
    def fit_gmm(
        self,
        person_n_components: Optional[int] = None,
        household_n_components: Optional[int] = None,
        auto_select: bool = True
    ):
        """训练GMM模型"""
        print("Training Person Activity Pattern GMM...")
        self.person_gmm.fit_from_features(
            self.person_features,
            auto_select_k=auto_select,
            n_components=person_n_components
        )
        
        print("\nTraining Household Activity Pattern GMM...")
        self.household_gmm.fit_from_features(
            self.household_features,
            auto_select_k=auto_select,
            n_components=household_n_components
        )
        
        return self
    
    def predict_patterns(
        self
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """预测模式标签和概率"""
        person_X = np.vstack([f.to_vector() for f in self.person_features])
        household_X = np.vstack([f.to_vector() for f in self.household_features])
        
        person_labels = self.person_gmm.predict(person_X)
        person_probs = self.person_gmm.predict_proba(person_X)
        
        household_labels = self.household_gmm.predict(household_X)
        household_probs = self.household_gmm.predict_proba(household_X)
        
        return person_labels, person_probs, household_labels, household_probs
    
    def get_pattern_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取模式embedding（用于后续decoder）"""
        # 使用GMM均值作为pattern embedding
        person_pattern_emb = self.person_gmm.gmm.means_
        household_pattern_emb = self.household_gmm.gmm.means_
        
        return person_pattern_emb, household_pattern_emb
    
    def summarize(self):
        """打印模式摘要"""
        print("=" * 60)
        print("PERSON ACTIVITY PATTERNS")
        print("=" * 60)
        print(f"Number of patterns: {self.person_gmm.best_n_components}")
        print("\nPattern Interpretations:")
        for idx, desc in self.person_gmm.interpret_patterns().items():
            weight = self.person_gmm.gmm.weights_[idx]
            print(f"  Pattern {idx}: {desc} (weight: {weight:.3f})")
        
        print("\n" + "=" * 60)
        print("HOUSEHOLD ACTIVITY PATTERNS")
        print("=" * 60)
        print(f"Number of patterns: {self.household_gmm.best_n_components}")
        print("\nPattern Interpretations:")
        for idx, desc in self.household_gmm.interpret_patterns().items():
            weight = self.household_gmm.gmm.weights_[idx]
            print(f"  Pattern {idx}: {desc} (weight: {weight:.3f})")


# ==================== 使用示例 ====================

def demo():
    """演示用法"""
    np.random.seed(42)
    
    # 模拟数据
    n_households = 100
    household_data = {}
    
    for hh_idx in range(n_households):
        hh_id = f"HH_{hh_idx:04d}"
        n_members = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
        
        members = []
        for m_idx in range(n_members):
            n_activities = np.random.randint(1, 7)
            
            # 生成模拟活动数据 [n_activities, 27]
            activities = np.zeros((6, 27))  # max 6 activities
            mask = np.zeros(6)
            
            for a_idx in range(n_activities):
                # 时间特征
                activities[a_idx, 0] = np.random.randn()  # start time z-score
                activities[a_idx, 1] = activities[a_idx, 0] + abs(np.random.randn() * 0.5)  # end time
                
                # 目的 one-hot
                purpose = np.random.choice(10)
                activities[a_idx, 2 + purpose] = 1
                
                # 方式 one-hot
                mode = np.random.choice(11)
                activities[a_idx, 12 + mode] = 1
                
                # 是否驾驶员
                is_driver = np.random.choice(2)
                activities[a_idx, 23 + is_driver] = 1
                
                # 是否联合出行
                is_joint = np.random.choice(2, p=[0.7, 0.3])
                activities[a_idx, 25 + is_joint] = 1
                
                mask[a_idx] = 1
            
            members.append({
                'person_id': f"{hh_id}_P{m_idx}",
                'activities': activities,
                'mask': mask
            })
        
        household_data[hh_id] = {
            'info': {
                'n_members': n_members,
                'n_workers': np.random.randint(0, n_members + 1),
                'n_students': np.random.randint(0, max(1, n_members - 1)),
                'n_vehicles': np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
            },
            'members': members
        }
    
    # 运行Pipeline
    pipeline = ActivityPatternPipeline(
        person_gmm_config={'n_components_range': (3, 8)},
        household_gmm_config={'n_components_range': (3, 6)}
    )
    
    # 提取特征
    print("Extracting features...")
    person_features, household_features = pipeline.extract_features(household_data)
    print(f"Extracted {len(person_features)} person features")
    print(f"Extracted {len(household_features)} household features")
    
    # 训练GMM
    print("\nFitting GMM models...")
    pipeline.fit_gmm(auto_select=True)
    
    # 预测
    person_labels, person_probs, hh_labels, hh_probs = pipeline.predict_patterns()
    
    # 摘要
    print("\n")
    pipeline.summarize()
    
    # 获取pattern embedding
    person_pattern_emb, hh_pattern_emb = pipeline.get_pattern_embeddings()
    print(f"\nPerson pattern embedding shape: {person_pattern_emb.shape}")
    print(f"Household pattern embedding shape: {hh_pattern_emb.shape}")
    
    return pipeline


if __name__ == "__main__":
    pipeline = demo()
