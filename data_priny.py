
import pandas as pd

from pathlib import Path


best_epoch = 50
best_tag = 'metrics/mAP50(B)_plane'
dic = {
    "":"",
    
}
def pr(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        

        # print(df.head())
        df = df.round(3)
        df.columns = df.columns.str.strip()

        if 'epoch' not in df.columns:
            print(df.columns)
            print("DataFrame中不存在'epoch'列，请检查列名。")
        else:
            print(f"获取前{best_epoch}最佳mAP50(B)")
            print(f"best 指标：{best_tag}")
            filtered_df = df[df['epoch'] <= best_epoch]
            max_row = filtered_df.loc[filtered_df[best_tag].idxmax()]

            print(csv_file_path)
            print("飞机only" + "\n" + f"|=={max_row['metrics/precision(B)_plane']}==| =={max_row['metrics/recall(B)_plane']}== | =={max_row['metrics/mAP50(B)_plane']}== | =={max_row['metrics/mAP50-95(B)_plane']}== |")
            print("飞机+车" + "\n" + f"|=={max_row['metrics/precision(B)']}==| =={max_row['metrics/recall(B)']}== | =={max_row['metrics/mAP50(B)']}== | =={max_row['metrics/mAP50-95(B)']}== |")
            
    except Exception as e: 
        print(f"读取或处理CSV文件时出错: {e}")


# 设置要遍历的目录路径
directory_path = Path('/home/xxf/xxf/code/yolov8-two-stream/runs')

# 遍历目录及其所有子目录中的所有CSV文件
for csv_file in directory_path.rglob('*.csv'):
    # print(csv_file)
    pr(csv_file)

# csv_file_path = "runs/30shot_vehicle_two/exp3/results.csv" 