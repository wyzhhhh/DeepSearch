
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, zscore, norm

import statsmodels.api as sm
from scipy.stats import norm

def score_graph(data,title):

    data_skewness = skew(data)  # 偏态
    data_kurtosis = kurtosis(data)  # 峰度

    # 绘制数据的直方图
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

    # 添加正态分布曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, np.mean(data), np.std(data))
    plt.plot(x, p, 'k', linewidth=2)

    title = f"{title} Fit results: Skewness = {data_skewness:.2f}, Kurtosis = {data_kurtosis:.2f}"
    plt.title(title)
    plt.show()

    # 计算Z分数
    data_z_scores = zscore(data)
    # print(f"Z-scores: {data_z_scores}")

    # 可以选择输出哪些是潜在的异常值，这里我们设定Z分数的阈值为2
    outliers = data[np.abs(data_z_scores) > 2]
    print(f"Potential outliers: {outliers}")

def TOPSIS_score(df,type,output_path):

    
    data = df[['impact_Normalized', 'relevance_score', 'cited_Normalized']].values

    norm_data = data / np.sqrt((data ** 2).sum(axis=0))
    weights = np.array([0.2, 0.6, 0.2])  # 假设的权重
    weighted_data = norm_data * weights
    ideal_best = np.max(weighted_data, axis=0)
    ideal_worst = np.min(weighted_data, axis=0)
    distance_to_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    distance_to_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))

    similarity_to_best = distance_to_worst / (distance_to_best + distance_to_worst)

    df['topsis_score'] = similarity_to_best
    return df
    #df.to_csv(output_path, index=False)  






def rsr(data, weight=None, threshold=None, full_rank=True):

    Result = pd.DataFrame()
    n, m = data.shape

    # 对原始数据编秩
    if full_rank:
        for i, X in enumerate(data.columns):
            Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]
            Result[f'R{str(i + 1)}：{X}'] = data.iloc[:, i].rank(method="dense")
    else:
        for i, X in enumerate(data.columns):
            Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]
            Result[f'R{str(i + 1)}：{X}'] = 1 + (n - 1) * (data.iloc[:, i].max() - data.iloc[:, i]) / (data.iloc[:, i].max() - data.iloc[:, i].min())

    # 计算秩和比
    weight = 1 / m if weight is None else np.array(weight) / sum(weight)
    Result['RSR'] = (Result.iloc[:, 1::2] * weight).sum(axis=1) / n
    Result['RSR_Rank'] = Result['RSR'].rank(ascending=False)

    # 绘制 RSR 分布表
    RSR = Result['RSR']
    RSR_RANK_DICT = dict(zip(RSR.values, RSR.rank().values))
    Distribution = pd.DataFrame(index=sorted(RSR.unique()))
    Distribution['f'] = RSR.value_counts().sort_index()
    Distribution['Σ f'] = Distribution['f'].cumsum()
    Distribution[r'\bar{R} f'] = [RSR_RANK_DICT[i] for i in Distribution.index]
    Distribution[r'\bar{R}/n*100%'] = Distribution[r'\bar{R} f'] / n
    Distribution.iat[-1, -1] = 1 - 1 / (4 * n)
    Distribution['Probit'] = 5 - norm.isf(Distribution.iloc[:, -1])

    # 计算回归方差并进行回归分析
    r0 = np.polyfit(Distribution['Probit'], Distribution.index, deg=1)
    print(sm.OLS(Distribution.index, sm.add_constant(Distribution['Probit'])).fit().summary())
    if r0[1] > 0:
        print(f"\n回归直线方程为：y = {r0[0]} Probit + {r0[1]}")
    else:
        print(f"\n回归直线方程为：y = {r0[0]} Probit - {abs(r0[1])}")

    # 代入回归方程并分档排序
    Result['Probit'] = Result['RSR'].apply(lambda item: Distribution.at[item, 'Probit'])
    Result['RSR Regression'] = np.polyval(r0, Result['Probit'])
    threshold = np.polyval(r0, [2, 4, 6, 8]) if threshold is None else np.polyval(r0, threshold)
    Result['Level'] = pd.cut(Result['RSR Regression'], threshold, labels=range(len(threshold) - 1, 0, -1))

    return Result, Distribution


def rsrAnalysis(data, file_name=None, **kwargs):
    Result, Distribution = rsr(data, **kwargs)
    file_name = 'RSR 分析结果报告.xlsx' if file_name is None else file_name + '.xlsx'
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        Result.to_excel(writer, sheet_name='综合评价结果')
        Result.sort_values(by='Level', ascending=False).to_excel(writer, sheet_name='分档排序结果')
        Distribution.to_excel(writer, sheet_name='RSR分布表')

    return Result, Distribution





"""



selected_data = rerank_citation_df[['Cited Number', 'relevance_score', 'impact_factor']]
result, distribution = rsrAnalysis(selected_data, file_name=f'./output/csv_download/{max_pmid}_rsr')

# 打印结果
print(result)
print(distribution)

result = result.sort_index()
result = result.reindex(data.index)
data['RSR'] = result['RSR']
data['Probit'] = result['Probit']
data.to_csv('./output/csv_download/29544768_updated_data.csv', index=False)


#change('./output/csv_download/31230860_rerank_jina.csv',type="a",output_path="./output/csv_download/31230860_topsis_new.csv")
#change("./output/csv_download/31230860_topsis_new.csv",type="b",output_path="./output/csv_download/31230860_topsis.csv")
#df = pd.read_csv('./output/csv_download/31230860_topsis.csv')
#f(df['final_score'],title="final_score")
#f(df['topsis_score'],title="topsis")
#f(df['topsis_new_score'],title="topsis_new")



df = pd.read_csv('./output/csv_download/29544768_updated_data.csv')
f(df['Probit'],title="RSR")
f(df['RSR'],title="rsr")
f(df['topsis_new_score'],title="TOPSIS")
"""