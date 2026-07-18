from pathlib import Path


# 公共路径参数，均使用相对路径
DATA_FILE = Path("docs") / "reference_formats.xlsx"
FIGURE_DIR = Path("outputs") / "figures"
TABLE_DIR = Path("outputs") / "tables"

# 问题 1.1 使用的工作表
STANDARD_SHEET_NAME = "Sheet1_标准实验数据"

# 中文字体候选，按本机可用字体自动匹配
CHINESE_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]

# 输出图像文件名
QUESTION1_FIGURE = FIGURE_DIR / "question1_1_linear_fit.png"
