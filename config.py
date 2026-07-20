from pathlib import Path


# 公共路径参数，均使用相对路径
DATA_FILE = Path("docs") / "reference_formats.xlsx"
FIGURE_DIR = Path("outputs") / "figures"
TABLE_DIR = Path("outputs") / "tables"

# 问题 1.1 使用的工作表
STANDARD_SHEET_NAME = "Sheet1_标准实验数据"

# 问题 1.2 使用的工作表
CONDITION_SHEETS = {
    "岩石工况": "Sheet2_岩石工况数据",
    "煤体工况": "Sheet3_煤体工况数据",
}

# R² 参考线
R2_REFERENCE_LINE = 0.99
MIN_LINEAR_POINTS = 3

# 问题 1.2 均为直径 20 mm 锚杆
QUESTION1_2_BOLT_DIAMETER_MM = 20.0

# 中文字体候选，按本机可用字体自动匹配
CHINESE_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]

# 输出图像文件名
QUESTION1_1_FIGURE = FIGURE_DIR / "question1_1_linear_fit.png"

QUESTION1_2_R2_CV_K_FIGURES = {
    "岩石工况": FIGURE_DIR / "question1_2_rock_r2_cv_k.png",
    "煤体工况": FIGURE_DIR / "question1_2_coal_r2_cv_k.png",
}

# 固定临界力矩后的最终过原点拟合
QUESTION1_2_FIXED_CRITICAL_TORQUES = {
    "岩石工况": 125.0,
    "煤体工况": 175.0,
}
QUESTION1_2_FIXED_FIT_FIGURES = {
    "岩石工况": FIGURE_DIR / "question1_2_rock_fixed_fit.png",
    "煤体工况": FIGURE_DIR / "question1_2_coal_fixed_fit.png",
}
QUESTION1_2_FIXED_RESIDUAL_FIGURES = {
    "岩石工况": FIGURE_DIR / "question1_2_rock_fixed_residuals.png",
    "煤体工况": FIGURE_DIR / "question1_2_coal_fixed_residuals.png",
}
