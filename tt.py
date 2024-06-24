import pandas as pd

# 读取CSV文件
df = pd.read_csv('runs/10_sar/exp2/results.csv')
df.columns = df.columns.str.replace(' ', '')
# print(df.columns)

# 假设我们要前k个epoch的数据
k = 100  # 请根据需要设置k值

# 选择前k个epoch的数据
df_k = df.head(k)
print(df_k)

best_map50 = df_k['metrics/mAP50(B)'].max()
best_row = df_k[df_k['metrics/mAP50(B)'] == best_map50].iloc[0]
best_map50 = best_row['metrics/mAP50(B)_plane']
best_precision = best_row['metrics/precision(B)_plane']
best_recall = best_row['metrics/recall(B)_plane']

# 计算F1分数
best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall)
best_map50 = round(best_map50, 3)
best_precision = round(best_precision, 3)
best_recall = round(best_recall, 3)
best_f1 = round(best_f1, 3)
print(f"Best mAP50_plane: {best_map50}")
print(f"Corresponding Precision_plane: {best_precision}")
print(f"Corresponding Recall_plane: {best_recall}")
print(f"Corresponding F1 Score_plane: {best_f1}")


# 找到metrics/mAP50(B)中的最大值
best_map50 = df_k['metrics/mAP50(B)'].max()
best_row = df_k[df_k['metrics/mAP50(B)'] == best_map50].iloc[0]
best_map50 = best_row['metrics/mAP50(B)']

# 找到对应的precision和recall
# best_row = df_k[df_k['metrics/mAP50(B)'] == best_map50].iloc[0]
best_precision = best_row['metrics/precision(B)']
best_recall = best_row['metrics/recall(B)']

# 计算F1分数
best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall)
best_map50 = round(best_map50, 3)
best_precision = round(best_precision, 3)
best_recall = round(best_recall, 3)
best_f1 = round(best_f1, 3)
print(f"Best mAP50: {best_map50}")
print(f"Corresponding Precision: {best_precision}")
print(f"Corresponding Recall: {best_recall}")
print(f"Corresponding F1 Score: {best_f1}")

