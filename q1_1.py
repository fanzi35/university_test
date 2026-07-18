import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 参数设置
# ============================================================

EXCEL_PATH = "A-附件.xlsx"

OUTPUT_FIGURE = "Sheet1_分组线性拟合图.png"
OUTPUT_EXCEL = "Sheet1_分组线性拟合结果.xlsx"


# ============================================================
# 2. Matplotlib中文显示
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. 读取第一个Sheet
# ============================================================

df = pd.read_excel(
    EXCEL_PATH,
    sheet_name=0
)

print("原始列名：")
print(df.columns.tolist())


# ============================================================
# 4. 清理列名
# ============================================================

def clean_column_name(column_name):
    """
    删除列名中的换行、空格和制表符。

    例如：
    '锚杆直径\\n/mm' -> '锚杆直径/mm'
    """
    return re.sub(
        r"\s+",
        "",
        str(column_name)
    )


df.columns = [
    clean_column_name(column)
    for column in df.columns
]

print("\n清理后的列名：")
print(df.columns.tolist())


# ============================================================
# 5. 自动识别三列
# ============================================================

diameter_col = None
torque_col = None
force_col = None

for column in df.columns:

    if "锚杆直径" in column:
        diameter_col = column

    elif "预紧力矩" in column:
        torque_col = column

    elif (
        "预紧力" in column
        and "预紧力矩" not in column
    ):
        force_col = column


if diameter_col is None:
    raise ValueError("没有找到“锚杆直径”列。")

if torque_col is None:
    raise ValueError("没有找到“预紧力矩”列。")

if force_col is None:
    raise ValueError("没有找到“预紧力”列。")


print("\n识别到的列：")
print("锚杆直径：", diameter_col)
print("预紧力矩：", torque_col)
print("预紧力：", force_col)


# ============================================================
# 6. 清洗数据
# ============================================================

# 原Excel中，每组锚杆直径只在第一行填写，
# 因此需要向下填充
df[diameter_col] = df[diameter_col].ffill()

# 转换为数值
for column in [
    diameter_col,
    torque_col,
    force_col
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# 删除缺失数据
df = df.dropna(
    subset=[
        diameter_col,
        torque_col,
        force_col
    ]
).copy()

# 按照直径、预紧力矩排序
df = df.sort_values(
    by=[
        diameter_col,
        torque_col
    ]
).reset_index(drop=True)


print("\n清洗后的数据：")
print(df)


# ============================================================
# 7. 定义最小二乘拟合函数
# ============================================================

def linear_least_squares(x, y):
    """
    最小二乘拟合线性函数：

        y = kx + c

    返回：
        k          斜率
        c          截距
        y_pred     拟合值
        r          Pearson相关系数
        r_squared  决定系数
        sse        残差平方和
        rmse       均方根误差
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        raise ValueError(
            "至少需要两个数据点才能进行线性拟合。"
        )

    # 最小二乘拟合
    k, c = np.polyfit(
        x,
        y,
        deg=1
    )

    y_pred = k * x + c

    # Pearson线性相关系数
    r = np.corrcoef(x, y)[0, 1]

    # 残差平方和
    sse = np.sum(
        (y - y_pred) ** 2
    )

    # 总离差平方和
    sst = np.sum(
        (y - np.mean(y)) ** 2
    )

    # 决定系数
    if sst == 0:
        r_squared = np.nan
    else:
        r_squared = 1 - sse / sst

    # 均方根误差
    rmse = np.sqrt(
        np.mean((y - y_pred) ** 2)
    )

    return {
        "k": k,
        "c": c,
        "y_pred": y_pred,
        "r": r,
        "r_squared": r_squared,
        "sse": sse,
        "rmse": rmse
    }


# ============================================================
# 8. 按照锚杆直径进行分组拟合
# ============================================================

diameters = sorted(
    df[diameter_col].unique()
)

fit_results = {}
summary_results = []

for diameter in diameters:

    group = df[
        df[diameter_col] == diameter
    ].copy()

    group = group.sort_values(
        torque_col
    )

    x = group[torque_col].to_numpy(
        dtype=float
    )

    y = group[force_col].to_numpy(
        dtype=float
    )

    result = linear_least_squares(
        x,
        y
    )

    fit_results[diameter] = {
        "x": x,
        "y": y,
        **result
    }

    summary_results.append({
        "锚杆直径/mm": diameter,
        "数据点数": len(x),
        "斜率k": result["k"],
        "截距c": result["c"],
        "线性相关系数r": result["r"],
        "决定系数R2": result["r_squared"],
        "残差平方和SSE": result["sse"],
        "均方根误差RMSE": result["rmse"]
    })


summary_df = pd.DataFrame(
    summary_results
)


# ============================================================
# 9. 输出每组拟合结果
# ============================================================

print("\n" + "=" * 80)
print("各锚杆直径的最小二乘线性拟合结果")
print("=" * 80)

for diameter in diameters:

    result = fit_results[diameter]

    k = result["k"]
    c = result["c"]
    r = result["r"]
    r_squared = result["r_squared"]

    print(f"\n锚杆直径：{diameter:g} mm")

    print(
        "拟合方程："
        f"P = {k:.8f}T "
        f"{c:+.8f}"
    )

    print(
        f"线性相关系数 r = {r:.8f}"
    )

    print(
        f"决定系数 R² = {r_squared:.8f}"
    )

    print(
        f"残差平方和 SSE = "
        f"{result['sse']:.8f}"
    )

    print(
        f"均方根误差 RMSE = "
        f"{result['rmse']:.8f}"
    )


print("\n" + "=" * 80)
print("拟合结果汇总")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 10. 绘制各组散点和拟合直线
# ============================================================

number_of_groups = len(diameters)

fig, axes = plt.subplots(
    nrows=1,
    ncols=number_of_groups,
    figsize=(6 * number_of_groups, 5.5),
    squeeze=False
)

axes = axes.flatten()


for ax, diameter in zip(
    axes,
    diameters
):

    result = fit_results[diameter]

    x = result["x"]
    y = result["y"]

    k = result["k"]
    c = result["c"]

    r = result["r"]
    r_squared = result["r_squared"]

    # 生成连续横坐标，使拟合直线更平滑
    x_plot = np.linspace(
        x.min(),
        x.max(),
        300
    )

    y_plot = k * x_plot + c

    # 绘制原始散点
    ax.scatter(
        x,
        y,
        s=65,
        marker="o",
        label="实验数据",
        zorder=5
    )

    # 绘制拟合直线
    ax.plot(
        x_plot,
        y_plot,
        linewidth=2.2,
        label=(
            "最小二乘拟合\n"
            f"$P={k:.4f}T"
            f"{c:+.4f}$\n"
            f"$r={r:.4f}$，"
            f"$R^2={r_squared:.4f}$"
        )
    )

    ax.set_title(
        f"锚杆直径 d={diameter:g} mm",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "预紧力矩 T / (N·m)",
        fontsize=12
    )

    ax.set_ylabel(
        "预紧力 P / kN",
        fontsize=12
    )

    ax.grid(
        linestyle="--",
        alpha=0.3
    )

    ax.legend(
        fontsize=10,
        loc="best"
    )


plt.suptitle(
    "不同锚杆直径下预紧力矩与预紧力的线性拟合",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. 保存拟合结果到Excel
# ============================================================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    # 保存拟合参数
    summary_df.to_excel(
        writer,
        sheet_name="线性拟合结果",
        index=False
    )

    # 保存清洗后的原始数据
    df.to_excel(
        writer,
        sheet_name="清洗后原始数据",
        index=False
    )

    # 保存各组的实际值和预测值
    for diameter in diameters:

        result = fit_results[diameter]

        detail_df = pd.DataFrame({
            "预紧力矩T/(N·m)": result["x"],
            "实际预紧力P/kN": result["y"],
            "拟合预紧力P/kN": result["y_pred"],
            "残差/kN": (
                result["y"] - result["y_pred"]
            )
        })

        sheet_name = (
            f"d={diameter:g}mm拟合明细"
        )[:31]

        detail_df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )


print("\n程序运行完成。")
print(f"拟合图已保存为：{OUTPUT_FIGURE}")
print(f"拟合结果已保存为：{OUTPUT_EXCEL}")