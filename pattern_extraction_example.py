"""
活动模式提取使用示例

展示如何使用family_individual_pattern_extraction_corrected.py
从原始数据中提取家庭和个人活动模式

使用方法：
1. 确保已运行数据处理.ipynb中的数据加载和预处理代码
2. 运行此脚本进行模式提取
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append(os.getcwd())

# from family_individual_pattern_extraction_corrected import (
#     ActivityPatternExtractor,
#     PatternExtractionConfig,
#     load_data_from_notebook
# )

from family_individual_pattern_extraction_with_progress import (
    ActivityPatternExtractor,
    PatternExtractionConfig,
    load_data_from_notebook
)

def pre_process():

    pd.set_option('display.max_columns', None)
    family2014 = pd.read_csv('数据/居民出行数据/2014/family_2014.csv',dtype=str)
    travel2014 = pd.read_csv('数据/居民出行数据/2014/midtable_2014.csv',dtype=str)
    familymember_2014 = pd.read_csv('数据/居民出行数据/2014/family_member_2014.csv',dtype=str)
    family2023 = pd.read_csv('数据/居民出行数据/2023/family_total_33169.csv',dtype=str)
    travel2023 = pd.read_csv('数据/居民出行数据/2023/midtable_total_33169.csv',dtype=str)
    familymember_2023 = pd.read_csv('数据/居民出行数据/2023/familymember_total_33169.csv',dtype=str)
    family_cluster = pd.read_csv('数据/family_cluster_improved.csv',dtype=str)
    cluster_profile = pd.read_csv('数据/cluster_profile_improved.csv',dtype=str)
    cluster_profile.columns
    cluster_profile.iloc[:,1:] = cluster_profile.iloc[:,1:].astype(float)
    ## 家庭变量筛选
    valid_member_number = familymember_2023.groupby('家庭编号').size().rename('家庭成员数量_real').reset_index()
    family2023 = pd.merge(family2023, valid_member_number, on='家庭编号', how='left')
    family2023 = family2023[family2023['家庭成员数量'].astype(int) == family2023['家庭成员数量_real']]
    valid_family = family2023[['家庭编号']]
    familymember_2023 = pd.merge(familymember_2023, valid_family, on='家庭编号', how='inner')
    family2023[['家庭成员数量']].value_counts()
    ## 家庭连续型变量
    family2023[['家庭成员数量','家庭工作人口数','机动车数量','脚踏自行车数量','电动自行车数量','摩托车数量','老年代步车数量']]
    have_student_family = familymember_2023[familymember_2023['职业'] == '14'].drop_duplicates(['家庭编号'])[['家庭编号']]
    have_student_family['have_student'] = 1
    family2023 = pd.merge(family2023, have_student_family, on='家庭编号', how='left').fillna({'have_student':0})
    ## 家庭离散型变量
    family2023[['have_student','家庭年收入']]
    family2023['家庭年收入'].isna().sum()
    ## 个人变量筛选
    familymember_2023['age'] = 2023 - familymember_2023['出生年份'].astype(int)
    familymember_2023['age_group'] = pd.cut(familymember_2023['age'], bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], right=False, labels=['0-5','6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90','91-95','96-100'])
    familymember_2023['age'].max() , familymember_2023['age'].min()
    # familymember_2023['age'] = (familymember_2023['age'] - familymember_2023['age'].min()) / (familymember_2023['age'].max() - familymember_2023['age'].min())
    ## 连续型变量
    familymember_2023[['age']]
    (familymember_2023[familymember_2023['关系']=='0']['age']).describe()
    familymember_2023.loc[familymember_2023['最高学历'].isna(),'最高学历'] = familymember_2023.loc[familymember_2023['最高学历'].isna(),'教育阶段']
    ## 离散型变量,这里的关系有点不太对，有的户主很小
    familymember_2023[['性别','是否有驾照','关系','最高学历','职业']]
    familymember_2023['是否有驾照'] = familymember_2023['是否有驾照'].fillna('0')
    import numpy as np
    ## 变量编码
    income_map = {'A':1, 'B':1, 'C':2, 'D':2, 'E':3, 'F':3, 'G':4, 'I':5, 'J':5, 'K':5}
    family2023['家庭年收入'] = family2023['家庭年收入'].map(income_map)
    familymember_2023['age_group'] = pd.cut(
        familymember_2023['age'], 
        bins=range(0, familymember_2023['age'].max() + 6, 5),
        labels=False 
    )


    familymember_2023['age_group'] = familymember_2023['age_group'].fillna(0)
    familymember_2023['age'] = familymember_2023['age_group']
    # relation_map = {'0':0, '17':1, '1':2, '2':2, '5':2, '6':2, '13':3, '14':3, '15':3, '16':3, '9':3, '10':3, '7':4, '8':4, '11':5, '12':5}
    # education_map = {'1':1, '2':1, '3':2, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7}
    # occupation_map = {'1':1, '2':1, '3':1, '4':2, '5':2, '6':3, '7':2, '8':3, '9':1, '10':4, '11':4, '12':4, '13':5, '14':6, '15':7, '16':8, '17':8, '18':1, '19':1, '20':8}

    # familymember_2023['关系'] = familymember_2023['关系'].map(relation_map)
    # familymember_2023['最高学历'] = familymember_2023['最高学历'].map(education_map)
    # familymember_2023['职业'] = familymember_2023['职业'].map(occupation_map)
    familymember_2023['关系'].value_counts().shape, familymember_2023['最高学历'].value_counts().shape, familymember_2023['职业'].value_counts().shape


    family2023
    # 活动链提取
    travelinfo = pd.read_csv('数据/居民出行数据/2023/midtable_total_33169.csv',dtype=str)
    travelinfo = pd.merge(travelinfo, family2023[['家庭编号']])
    ## 活动链太大的去除掉
    travel_num_the = travelinfo.groupby(['家庭编号','家庭成员编号']).size().rename('出行次数').reset_index()
    more_travel = travel_num_the[travel_num_the['出行次数']>6][['家庭编号']].drop_duplicates()
    family2023 = family2023[-family2023['家庭编号'].isin(more_travel['家庭编号'])]
    travelinfo = pd.merge(travelinfo, family2023[['家庭编号']])
    familymember_2023 = pd.merge(familymember_2023, family2023[['家庭编号']])
    ## 没有出行的家庭
    have_travel_family = travelinfo[['家庭编号']].value_counts().reset_index()[['家庭编号']]
    family2023 = family2023[family2023['家庭编号'].isin(have_travel_family['家庭编号'])]
    travelinfo = pd.merge(travelinfo, family2023[['家庭编号']])
    familymember_2023 = pd.merge(familymember_2023, family2023[['家庭编号']])
    ## 目的、方式、时间、车辆、陪同
    activityinfo = travelinfo[['家庭编号','家庭成员编号','出行序号','出行目的','交通方式的编号','ModelMode','出发时间1小时时间段','到达时间1小时时间段','是驾驶员还是乘客','是否和家庭成员的联合出行']]
    ## 1:步行;2:自行车、电动自行车；3:公交、地铁；4:小汽车、出租车；5:其他
    ## 1:步行、自行车、电动自行车；2:公交、地铁；3:小汽车、出租车；4:其他
    ## 1:步行；2:公交；3:地铁；4:自行车；5:电动自行车；6:小汽车；7:其他机动车；8:班车类； 9:出租 ;10:摩托车; 11:其他

    class_map2023 = {
        "1": "1",
        "2.1": "6",
        "2.2": "7",
        "2.3": "7",
        "3": "7",
        "4": "7",
        "5": "10",
        "6": "3",
        "7": "2",
        "8": "9",
        "9": "9",
        "10": "8",
        "11": "8",
        "12": "8",
        "13": "4",
        "14": "4",
        "15": "4",
        "16": "5",
        "17": "11",
        "18": "11",
        "19": "11"
    }

    activityinfo['是驾驶员还是乘客'] = activityinfo['是驾驶员还是乘客'].fillna('0')
    activityinfo['ModelMode'] = activityinfo['交通方式的编号'].map(class_map2023)
    #2023数据
    # 1：工作
    # 2：外出就餐
    # 3：公务外出
    # 4：上学/校外托管
    # 5：探亲访友
    # 7：休闲娱乐健身
    # 8：个人事务
    # 10：接送人
    # 13：购物
    # 14：其他
    # 15：回另一居住地
    # 21：下班回家
    # 22：放学回家
    # 23：个人事务回家
    # 24：其他回家

    # 1：在家（15,21,22,23,24）
    # 2: 工作（1）
    # 3：上学（4）
    # 4：公务外出类（3）
    # 5：外出吃饭
    # 6：探亲访友
    # 7：休闲娱乐
    # 8：购物
    # 9：接送人
    # 10：其他（8，14）
    def get_activity_type2023(r):
        if r in ['15','21','22','23','24']:
            return '1'
        elif r == '1':
            return '2'
        elif r == '4':
            return '3'
        elif r == '3':
            return '4'
        elif r == '2':
            return '5'
        elif r == '5':
            return '6'
        elif r == '7':
            return '7'
        elif r == '13':
            return '8'
        elif r in ['10']:
            return '9'
        elif r in ['8', '14']:
            return '10'
    activityinfo['ActivityType'] = activityinfo['出行目的'].apply(get_activity_type2023)
    return family2023, familymember_2023, activityinfo

def main():
    """主函数：演示完整的模式提取流程"""
    
    print("=" * 60)
    print("家庭和个人活动模式提取示例")
    print("=" * 60)
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载原始数据")
    print("-" * 30)
    
    # 尝试从notebook环境加载数据
    try:
        # 如果在jupyter环境中，尝试获取全局变量
        family2023, familymember_2023, activityinfo = pre_process()
                
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保已运行数据处理.ipynb中的相关代码")
        return
    
    print(f"数据规模:")
    print(f"  - 家庭数: {len(family2023)}")
    print(f"  - 成员数: {len(familymember_2023)}")
    print(f"  - 活动记录数: {len(activityinfo)}")
    
    # 步骤2: 配置模式提取器
    print("\n步骤2: 配置模式提取器")
    print("-" * 30)
    
    config = PatternExtractionConfig(
        max_family_size=8,
        max_activities=6,
        person_n_components_range=(2, 250),
        household_n_components_range=(2, 125),
        random_state=42
    )
    
    extractor = ActivityPatternExtractor(config)
    print("模式提取器创建成功")
    
    # 步骤3: 准备训练数据（使用子集以节省时间）
    print("\n步骤3: 准备训练数据")
    print("-" * 30)
    
    # 选择前500个家庭作为示例
    n_families = len(family2023)
    sample_families = family2023['家庭编号'].unique()[:n_families]
    
    train_family = family2023[family2023['家庭编号'].isin(sample_families)].copy()
    train_member = familymember_2023[familymember_2023['家庭编号'].isin(sample_families)].copy()
    train_activity = activityinfo[activityinfo['家庭编号'].isin(sample_families)].copy()
    
    print(f"训练数据规模:")
    print(f"  - 家庭数: {len(train_family)}")
    print(f"  - 成员数: {len(train_member)}")
    print(f"  - 活动记录数: {len(train_activity)}")
    
    # 步骤4: 训练模式识别模型
    print("\n步骤4: 训练模式识别模型")
    print("-" * 30)
    
    try:
        extractor.fit_from_raw_data(
            train_family, train_member, train_activity,
            auto_select_k=True
        )
        print("模型训练完成")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return
    
    # 步骤5: 提取活动模式
    print("\n步骤5: 提取活动模式")
    print("-" * 30)
    
    try:
        family_patterns, individual_patterns, household_mapping = extractor.extract_all_patterns_from_raw(
            train_family, train_member, train_activity
        )
        
        print(f"模式提取完成:")
        print(f"  - 家庭模式矩阵: {family_patterns.shape} (批次 × 模式概率)")
        print(f"  - 个人模式矩阵: {individual_patterns.shape} (批次 × 成员 × 模式概率)")
        
    except Exception as e:
        print(f"模式提取过程中出错: {e}")
        return
    
    # 步骤6: 分析模式结果
    print("\n步骤6: 分析模式结果")
    print("-" * 30)
    
    # 获取模式解释
    person_interp, household_interp = extractor.get_pattern_interpretations()
    
    print("\n个人活动模式:")
    for i, desc in person_interp.items():
        weight = extractor.pattern_pipeline.person_gmm.gmm.weights_[i]
        print(f"  模式 {i}: {desc} (权重: {weight:.3f})")
    
    print("\n家庭活动模式:")
    for i, desc in household_interp.items():
        weight = extractor.pattern_pipeline.household_gmm.gmm.weights_[i]
        print(f"  模式 {i}: {desc} (权重: {weight:.3f})")
    
    # 模式分布统计
    print("\n模式分布统计:")
    
    # 家庭模式分布
    family_labels = np.argmax(family_patterns, axis=1)
    unique_f, counts_f = np.unique(family_labels, return_counts=True)
    print("\n家庭模式分布:")
    for pattern_id, count in zip(unique_f, counts_f):
        prop = count / len(family_labels)
        print(f"  模式 {pattern_id}: {count} 个家庭 ({prop:.1%})")
    
    # 个人模式分布
    valid_individual = individual_patterns[individual_patterns.sum(axis=2) > 0]
    if len(valid_individual) > 0:
        individual_labels = np.argmax(valid_individual, axis=1)
        unique_i, counts_i = np.unique(individual_labels, return_counts=True)
        print("\n个人模式分布:")
        for pattern_id, count in zip(unique_i, counts_i):
            prop = count / len(individual_labels)
            print(f"  模式 {pattern_id}: {count} 个个人 ({prop:.1%})")
    
    # 步骤7: 保存结果
    print("\n步骤7: 保存结果")
    print("-" * 30)
    
    # 保存模型
    extractor.save_model("gmm/activity_pattern_extractor.pkl")
    
    # 保存模式矩阵
    np.save("gmm/family_patterns_demo.npy", family_patterns)
    np.save("gmm/individual_patterns_demo.npy", individual_patterns)
    
    # 保存映射关系
    import json
    with open("gmm/household_mapping_demo.json", "w", encoding='utf-8') as f:
        json.dump(household_mapping, f, ensure_ascii=False, indent=2)
    
    print("结果保存完成:")
    print("  - activity_pattern_extractor.pkl: 训练好的模型")
    print("  - family_patterns_demo.npy: 家庭模式矩阵 [B, P]")
    print("  - individual_patterns_demo.npy: 个人模式矩阵 [B, M, P]")
    print("  - household_mapping_demo.json: 家庭ID映射")
    
    # 步骤8: 演示如何使用结果
    print("\n步骤8: 使用示例")
    print("-" * 30)
    
    # 示例：查看第一个家庭的模式
    if len(household_mapping) > 0:
        first_hh_id = list(household_mapping.keys())[0]
        first_hh_idx = household_mapping[first_hh_id]
        
        print(f"\n示例：家庭 {first_hh_id} 的模式分析")
        print(f"家庭模式概率: {family_patterns[first_hh_idx]}")
        print(f"主要家庭模式: {np.argmax(family_patterns[first_hh_idx])}")
        
        # 该家庭成员的模式
        member_patterns = individual_patterns[first_hh_idx]
        valid_members = member_patterns[member_patterns.sum(axis=1) > 0]
        
        print(f"有效成员数: {len(valid_members)}")
        for i, member_pattern in enumerate(valid_members):
            main_pattern = np.argmax(member_pattern)
            print(f"  成员 {i}: 主要模式 {main_pattern} (概率: {member_pattern[main_pattern]:.3f})")
    
    print("\n" + "=" * 60)
    print("模式提取演示完成！")
    print("=" * 60)
    
    return extractor, family_patterns, individual_patterns


if __name__ == "__main__":
    # 运行示例
    result = main()
    
    if result:
        print("\n演示成功完成，可以使用提取的模式进行后续分析")