from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CONDITION_SHEETS,
    DATA_FILE,
    MIN_LINEAR_POINTS,
    QUESTION1_1_FIGURE,
    QUESTION1_2_BOLT_DIAMETER_MM,
    QUESTION1_2_FIXED_CRITICAL_TORQUES,
    QUESTION1_2_FIXED_FIT_FIGURES,
    QUESTION1_2_FIXED_RESIDUAL_FIGURES,
    QUESTION1_2_R2_CV_K_FIGURES,
    R2_REFERENCE_LINE,
    STANDARD_SHEET_NAME,
    FIGURE_DIR,
)
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


def load_condition_data(sheet_name: str, condition: str, path: Path = DATA_FILE) -> pd.DataFrame:
    """读取工况实测数据，并转换为长表。"""
    raw = read_xlsx_sheet(path, sheet_name)
    torque_col = raw.columns[0]
    point_cols = list(raw.columns[1:6])
    raw = raw[[torque_col, *point_cols]].copy()
    raw = raw.rename(columns={torque_col: "torque_nm"})
    raw["torque_nm"] = pd.to_numeric(raw["torque_nm"], errors="coerce")

    long_data = raw.melt(
        id_vars="torque_nm",
        value_vars=point_cols,
        var_name="point",
        value_name="preload_kn",
    )
    long_data["preload_kn"] = pd.to_numeric(long_data["preload_kn"], errors="coerce")
    long_data = long_data.dropna(subset=["torque_nm", "preload_kn"]).reset_index(drop=True)
    long_data["point"] = long_data["point"].str.extract(r"(测点\d+-\d+)", expand=False).fillna(long_data["point"])
    long_data.insert(0, "condition", condition)
    return long_data


def average_condition_data(data: pd.DataFrame) -> pd.DataFrame:
    """按力矩计算 5 个测点预紧力均值。"""
    averaged = (
        data.groupby(["condition", "torque_nm"], as_index=False)["preload_kn"]
        .mean()
        .sort_values("torque_nm")
        .reset_index(drop=True)
    )
    return averaged


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


def proportional_fit_metrics(
    preload_kn: np.ndarray,
    torque_nm: np.ndarray,
    diameter_mm: float = QUESTION1_2_BOLT_DIAMETER_MM,
) -> dict[str, float]:
    """计算正比例拟合的斜率、决定系数和等效扭矩系数 CV_K。"""
    fit = fit_torque_preload(preload_kn, torque_nm)
    preload = np.asarray(preload_kn, dtype=float)
    torque = np.asarray(torque_nm, dtype=float)
    predicted = fit.slope * preload
    rss = float(np.sum((torque - predicted) ** 2))
    tss = float(np.sum((torque - np.mean(torque)) ** 2))
    rss_ratio = 0.0 if tss == 0 else rss / tss
    equivalent_torque_coefficient = torque / (preload * diameter_mm)
    # CV_K 使用样本标准差，并按百分比输出
    equivalent_coefficient_cv = float(
        np.std(equivalent_torque_coefficient, ddof=1)
        / np.mean(equivalent_torque_coefficient)
        * 100
    )
    return {
        "slope": fit.slope,
        "r2": fit.r2,
        "rss": rss,
        "rss_ratio": rss_ratio,
        "equivalent_torque_coefficient_cv": equivalent_coefficient_cv,
    }


def build_truncation_metrics(data: pd.DataFrame, min_points: int = MIN_LINEAR_POINTS) -> pd.DataFrame:
    """先按同一力矩求平均预紧力，再逐步截断计算 R² 与 CV_K。"""
    averaged = average_condition_data(data)
    condition = str(averaged["condition"].iloc[0])
    torque_values = np.sort(averaged["torque_nm"].unique())
    records = []

    for start_index in range(0, len(torque_values) - min_points + 1):
        start_torque = float(torque_values[start_index])
        suffix = averaged[averaged["torque_nm"] >= start_torque].sort_values("torque_nm")
        metrics = proportional_fit_metrics(suffix["preload_kn"].to_numpy(), suffix["torque_nm"].to_numpy())
        records.append(
            {
                "condition": condition,
                "start_index": start_index,
                "start_torque_nm": start_torque,
                **metrics,
            }
        )

    return pd.DataFrame(records)


def build_fixed_critical_fit(data: pd.DataFrame, critical_torque_nm: float) -> tuple[FitResult, pd.DataFrame]:
    """使用固定临界力矩后的均值点拟合过原点直线。"""
    averaged = average_condition_data(data)
    fitted_data = averaged[averaged["torque_nm"] >= critical_torque_nm].sort_values("torque_nm").reset_index(drop=True)
    fit = fit_torque_preload(fitted_data["preload_kn"].to_numpy(), fitted_data["torque_nm"].to_numpy())
    fitted_data = fitted_data.copy()
    fitted_data["predicted_torque_nm"] = fit.slope * fitted_data["preload_kn"]
    fitted_data["predicted_preload_kn"] = fitted_data["torque_nm"] / fit.slope
    fitted_data["force_residual_kn"] = fitted_data["preload_kn"] - fitted_data["predicted_preload_kn"]
    return fit, fitted_data


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


def plot_question1_1_fit(data: pd.DataFrame, results: pd.DataFrame, output_path: Path) -> None:
    """绘制三种直径下 T 与 P 的过原点拟合图像。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    for axis, (_, row) in zip(axes, results.iterrows()):
        diameter = row["diameter_mm"]
        group = data[data["diameter_mm"] == diameter].sort_values("preload_kn")
        max_preload = float(group["preload_kn"].max())
        x_values = np.linspace(0, max_preload * 1.05, 100)

        axis.scatter(group["preload_kn"], group["torque_nm"], color="#1f77b4", label="实验点")
        axis.plot(x_values, row["slope"] * x_values, color="#d62728", linewidth=1.6, label="过原点拟合")
        axis.text(
            0.95,
            0.08,
            f"d = {diameter:.0f} mm\nT = {row['slope']:.3f}P\nR2 = {row['r2']:.4f}\nK = {row['k']:.4f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
        axis.set_xlabel("预紧力 P / kN")
        axis.set_ylabel("预紧力矩 T / N·m")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=8)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_r2_cv_k_curves(
    metrics: pd.DataFrame,
    output_path: Path,
    critical_torque_nm: float | None = None,
) -> None:
    """绘制 R² 和等效扭矩系数 CV_K 随截断起始力矩变化的曲线。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    group = metrics.sort_values("start_torque_nm")
    axes[0].plot(group["start_torque_nm"], group["r2"], marker="o", label="五测点均值")
    axes[0].axhline(R2_REFERENCE_LINE, color="#d62728", linestyle="--", linewidth=1.2, label="99%参考线")
    axes[0].set_xlabel("截断起始力矩 / N·m")
    axes[0].set_ylabel("决定系数 R²")

    axes[1].plot(
        group["start_torque_nm"],
        group["equivalent_torque_coefficient_cv"],
        marker="o",
        label="五测点均值",
    )
    # CV_K 图从 0 开始，便于临界点虚线明确指向 x 轴
    axes[1].set_ylim(bottom=0)
    if critical_torque_nm is not None:
        critical_rows = group[np.isclose(group["start_torque_nm"], critical_torque_nm)]
        if not critical_rows.empty:
            critical_row = critical_rows.iloc[0]
            critical_x = float(critical_row["start_torque_nm"])
            critical_y = float(critical_row["equivalent_torque_coefficient_cv"])
            axes[1].scatter(
                [critical_x],
                [critical_y],
                color="#d62728",
                zorder=5,
                label="固定临界点",
            )
            axes[1].vlines(
                critical_x,
                0.0,
                critical_y,
                colors="black",
                linestyles="--",
                linewidth=1.2,
                alpha=0.9,
            )
            axes[1].text(
                critical_x,
                0.0,
                f"{critical_x:.0f}",
                ha="center",
                va="top",
                color="black",
                fontsize=9,
                clip_on=False,
            )
    axes[1].set_xlabel("截断起始力矩 / N·m")
    axes[1].set_ylabel("等效扭矩系数变异系数 CV_K / %")

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=8)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fixed_critical_fit(
    fit: FitResult,
    fitted_data: pd.DataFrame,
    critical_torque_nm: float,
    output_path: Path,
) -> None:
    """绘制固定临界点后的过原点拟合直线。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)

    max_preload = float(fitted_data["preload_kn"].max())
    x_values = np.linspace(0, max_preload * 1.05, 100)
    axis.scatter(fitted_data["preload_kn"], fitted_data["torque_nm"], color="#1f77b4", label="均值点")
    axis.plot(x_values, fit.slope * x_values, color="#d62728", linewidth=1.6, label="过原点拟合")
    axis.text(
        0.95,
        0.08,
        f"T = {fit.slope:.3f}P\nR2 = {fit.r2:.4f}\n临界力矩 = {critical_torque_nm:.0f} N·m",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    axis.set_xlabel("预紧力 P / kN")
    axis.set_ylabel("预紧力矩 T / N·m")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fixed_critical_residuals(fitted_data: pd.DataFrame, output_path: Path) -> None:
    """绘制固定临界点后力残差相对零线的分布。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)

    axis.axhline(0, color="#d62728", linestyle="--", linewidth=1.2, label="零残差线")
    axis.scatter(
        fitted_data["torque_nm"],
        fitted_data["force_residual_kn"],
        color="#1f77b4",
        label="力残差",
    )
    max_abs_residual = float(np.max(np.abs(fitted_data["force_residual_kn"])))
    residual_limit = max(2.0, max_abs_residual * 2.5)
    axis.set_ylim(-residual_limit, residual_limit)
    axis.set_xlabel("预紧力矩 T / N·m")
    axis.set_ylabel("力残差 / kN")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_question1_1(output_path: Path = QUESTION1_1_FIGURE) -> pd.DataFrame:
    """执行问题 1.1，生成三种直径的过原点拟合图。"""
    data = load_standard_data()
    results = build_fit_results(data)
    plot_question1_1_fit(data, results, output_path)
    return results


def run_question1_2() -> dict[str, pd.DataFrame]:
    """执行问题 1.2，生成 R²/CV_K、固定临界点拟合和残差图。"""
    ensure_dir(FIGURE_DIR)
    metrics_by_condition = {}

    for condition, sheet_name in CONDITION_SHEETS.items():
        raw_data = load_condition_data(sheet_name, condition)
        data = average_condition_data(raw_data)
        metrics = build_truncation_metrics(data)
        metrics_by_condition[condition] = metrics
        critical_torque = QUESTION1_2_FIXED_CRITICAL_TORQUES[condition]
        plot_r2_cv_k_curves(
            metrics,
            QUESTION1_2_R2_CV_K_FIGURES[condition],
            critical_torque_nm=critical_torque,
        )
        fit, fitted_data = build_fixed_critical_fit(data, critical_torque)
        plot_fixed_critical_fit(fit, fitted_data, critical_torque, QUESTION1_2_FIXED_FIT_FIGURES[condition])
        plot_fixed_critical_residuals(fitted_data, QUESTION1_2_FIXED_RESIDUAL_FIGURES[condition])

    return metrics_by_condition


def format_generation_header() -> str:
    """生成兼容 Windows 控制台编码的输出标题。"""
    return "问题1.2 R2/CV_K、固定临界点拟合和残差图像已生成："


def main() -> None:
    fit_results = run_question1_1()
    metrics_by_condition = run_question1_2()

    print("问题1.1 过原点拟合图像已生成：")
    for _, row in fit_results.iterrows():
        print(format_console_result(row))
    print(format_generation_header())
    for condition, metrics in metrics_by_condition.items():
        min_row = metrics.loc[metrics["equivalent_torque_coefficient_cv"].idxmin()]
        print(
            f"{condition}: CV_K最小截断起始力矩={min_row['start_torque_nm']:.0f} N·m, "
            f"CV_K={min_row['equivalent_torque_coefficient_cv']:.2f}%"
        )


if __name__ == "__main__":
    main()
