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

# ==================== 问题四：有效性与可靠性验证 ====================
# 论文中的模量比与完整性折减系数基准值
QUESTION4_MODULUS_RATIO = 300.0
QUESTION4_INTEGRITY_FACTOR = 1.0

# 论文式（49）的安全裕度系数 gamma：T_opt = gamma * Kd * min(P1, P2, P3)
QUESTION4_SAFETY_MARGIN_FACTOR = 0.80

# 单参数扰动幅度
QUESTION4_SINGLE_PERTURBATIONS = (-0.20, -0.10, 0.10, 0.20)

# 多参数随机扰动设置
QUESTION4_MULTI_PERTURBATION_LEVELS = (0.10, 0.20)
QUESTION4_MONTE_CARLO_SAMPLES = 5000
QUESTION4_RANDOM_SEED = 20260720

# 最坏情况：在所有参数的 ±20% 超矩形顶点中枚举最不利组合
QUESTION4_WORST_CASE_LEVEL = 0.20

# 最坏情况的稳健建议力矩同样使用 QUESTION4_SAFETY_MARGIN_FACTOR，
# 避免再引入第二个含义重复的裕度系数。

# 问题四输出路径
QUESTION4_VALIDITY_FIGURE = FIGURE_DIR / "question4_validity_constraints.png"
QUESTION4_SINGLE_PERTURBATION_FIGURE = FIGURE_DIR / "question4_single_parameter_sensitivity.png"
QUESTION4_MONTE_CARLO_FIGURE = FIGURE_DIR / "question4_multi_parameter_monte_carlo.png"
QUESTION4_WORST_CASE_FIGURE = FIGURE_DIR / "question4_worst_case_comparison.png"
QUESTION4_CONSTRAINT_CURVES_FIGURE = FIGURE_DIR / "question4_constraint_curves.png"

QUESTION4_VALIDITY_TABLE = TABLE_DIR / "question4_validity.csv"
QUESTION4_SINGLE_PERTURBATION_TABLE = TABLE_DIR / "question4_single_parameter_perturbation.csv"
QUESTION4_MONTE_CARLO_SUMMARY_TABLE = TABLE_DIR / "question4_monte_carlo_summary.csv"
QUESTION4_MONTE_CARLO_WORST_TABLE = TABLE_DIR / "question4_monte_carlo_worst_samples.csv"
QUESTION4_WORST_CASE_TABLE = TABLE_DIR / "question4_worst_case.csv"