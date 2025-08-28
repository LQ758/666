import pandas as pd
import random

def load_custom_data(file_path):
    """加载自定义练习数据（.txt 或 .csv）"""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({"sentence": lines})
    return df.to_dict(orient="records")

def get_random_custom_sentence(data_records):
    """从自定义数据中随机选择一条句子"""
    return random.choice(data_records)