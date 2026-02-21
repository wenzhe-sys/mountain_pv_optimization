# config/paths.py
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# 模块结果目录
MODULE1_RESULTS_DIR = os.path.join(RESULTS_DIR, 'module1')
MODULE2_RESULTS_DIR = os.path.join(RESULTS_DIR, 'module2')
MODULE3_RESULTS_DIR = os.path.join(RESULTS_DIR, 'module3')

# 可视化目录
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualization')

# 确保目录存在
os.makedirs(MODULE1_RESULTS_DIR, exist_ok=True)
os.makedirs(MODULE2_RESULTS_DIR, exist_ok=True)
os.makedirs(MODULE3_RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)