from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_FILE, QUESTION1_FIGURE, STANDARD_SHEET_NAME
from utils import ensure_dir, read_xlsx_sheet, setup_chinese_font


@dataclass(frozen=True)
class FitResult:
    slope: float
    intercept: float
    r2: float


def load_standard_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """读取标准实验数据，并整理为统一字段。"""
    raw = read_xlsx_sheet(path, STANDARD_SHEET_NAME)
    raw = raw.rename(
        columns={
            raw.columns[0]: "diameter_mm",
            raw.columns[1]: "torque_nm",
            raw.columns[2]: "preload_kn",
        }
    )

    # Excel 中相同直径只在首行填写，需要向下填充
    raw["diameter_mm"] = pd.to_numeric(raw["diameter_mm"], errors="coerce").ffill()
    raw["torque_nm"] = pd.to_numeric(raw["torque_nm"], errors="coerce")
    raw["preload_kn"] = pd.to_numeric(raw["preload_kn"], errors="coerce")
    return raw.dropna(subset=["diameter_mm", "torque_nm", "preload_kn"]).reset_index(drop=True)


def fit_torque_preload(preload_kn: np.ndarray, torque_nm: np.ndarray) -> FitResult:
    """使用过原点最小二乘法拟合 T = aP，并计算决定系数。"""
    preload = np.asarray(preload_kn, dtype=float)
    torque = np.asarray(torque_nm, dtype=float)
    slope = np.sum(preload * torque) / np.sum(preload**2)
    intercept = 0.0
    predicted = slope * preload

    residual_sum = np.sum((torque - predicted) ** 2)
    total_sum = np.sum((torque - np.mean(torque)) ** 2)
    r2 = 1.0 if total_sum == 0 else 1.0 - residual_sum / total_sum
    return FitResult(slope=float(slope), intercept=float(intercept), r2=float(r2))


def calculate_torque_coefficient(slope: float, diameter_mm: float) -> float:
    """由 T = K * P * d 得到 K = a / d。"""
    return float(slope) / float(diameter_mm)


def build_fit_results(data: pd.DataFrame) -> pd.DataFrame:
    """按锚杆直径分别拟合，并汇总模型参数。"""
    records = []
    for diameter, group in data.groupby("diameter_mm"):
        fit = fit_torque_preload(group["preload_kn"].to_numpy(), group["torque_nm"].to_numpy())
        records.append(
            {
                "diameter_mm": float(diameter),
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r2": fit.r2,
                "k": calculate_torque_coefficient(fit.slope, diameter),
            }
        )
    return pd.DataFrame(records).sort_values("diameter_mm").reset_index(drop=True)


def format_linear_equation(slope: float, intercept: float) -> str:
    """格式化正比例关系式。"""
    return f"T = {slope:.3f}P"


def format_console_result(row: dict | pd.Series) -> str:
    """生成兼容 Windows 控制台编码的结果文本。"""
    return (
        f"d={row['diameter_mm']:.0f}mm, "
        f"{format_linear_equation(row['slope'], row['intercept'])}, "
        f"R2={row['r2']:.4f}, K={row['k']:.4f}"
    )


def plot_fit_results(data: pd.DataFrame, results: pd.DataFrame, output_path: Path = QUESTION1_FIGURE) -> None:
    """绘制三种直径下的线性拟合图像。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)

    diameters = results["diameter_mm"].tolist()
    fig, axes = plt.subplots(1, len(diameters), figsize=(15, 4.8), constrained_layout=True)
    if len(diameters) == 1:
        axes = [axes]

    for axis, diameter in zip(axes, diameters):
        group = data[data["diameter_mm"] == diameter]
        result = results[results["diameter_mm"] == diameter].iloc[0]
        x = group["preload_kn"].to_numpy()
        y = group["torque_nm"].to_numpy()
        # 正比例拟合要求拟合线经过原点
        x_line = np.linspace(0, x.max(), 100)
        y_line = result["slope"] * x_line

        axis.scatter(x, y, color="#1f77b4", label="实验数据")
        axis.plot(x_line, y_line, color="#d62728", linewidth=2, label="最小二乘拟合")
        axis.set_xlabel("预紧力 P / kN")
        axis.set_ylabel("预紧力矩 T / N·m")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(loc="lower right")

        equation = (
            f"d = {diameter:.0f} mm\n"
            f"{format_linear_equation(result['slope'], result['intercept'])}\n"
            f"R² = {result['r2']:.4f}\n"
            f"K = {result['k']:.4f}"
        )
        axis.text(
            0.04,
            0.96,
            equation,
            transform=axis.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
        )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    data = load_standard_data()
    results = build_fit_results(data)
    plot_fit_results(data, results)

    print("问题1.1拟合结果：")
    for _, row in results.iterrows():
        print(format_console_result(row))
    print(f"图像已保存：{QUESTION1_FIGURE}")


if __name__ == "__main__":
    main()
