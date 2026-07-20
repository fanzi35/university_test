"""问题 3.2：工况 A、B 无钢带条件下约束曲线。

图中包含：
1. 锚杆杆体屈服约束；
2. 锚固系统约束；
3. 螺纹段屈服（拉伸—弯曲—扭转复合应力）约束；
4. 围岩压陷约束；
5. 四类约束的综合下包络线；
6. 问题二模型在不考虑偏心距时得到的最大允许预紧力矩基准线。

单位约定：
- 预紧力 P：kN
- 预紧力矩 T：N·m
- 长度 d、d_e、e、b、h0：mm
- 弹性模量 E：GPa
- 应力：MPa

由于 1 kN·mm = 1 N·m，关系 T = KdP 可直接使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


# ============================================================
# 1. 公共参数
# ============================================================

ALLOWABLE_STRESS_MPA = 500.0       # [σ]，MPa；附录 3 取安全系数 n=1
THREAD_TORQUE_RATIO = 0.09         # K_s
EFFECTIVE_DIAMETER_RATIO = 0.85    # d_e = 0.85d
ECCENTRIC_STEP_MM = 0.1            # 偏心距计算步长

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "question3_2_updated_outputs"


@dataclass(frozen=True)
class WorkingCondition:
    """单个工况的地质与支护参数。"""

    name: str
    short_name: str

    bolt_diameter_mm: float
    yield_load_kn: float
    hole_diameter_mm: float
    bond_length_mm: float
    elastic_modulus_gpa: float
    poisson_ratio: float
    tray_side_mm: float
    max_indent_mm: float
    influence_depth_mm: float

    torque_conversion: float       # Kd，单位 N·m/kN
    max_eccentricity_mm: float


CONDITIONS = (
    WorkingCondition(
        name="工况 A：岩层锚杆",
        short_name="A",
        bolt_diameter_mm=20.0,
        yield_load_kn=170.0,
        hole_diameter_mm=28.0,
        bond_length_mm=800.0,
        elastic_modulus_gpa=15.0,
        poisson_ratio=0.25,
        tray_side_mm=150.0,
        max_indent_mm=1.0,
        influence_depth_mm=900.0,
        torque_conversion=4.272,
        max_eccentricity_mm=25.0,
    ),
    WorkingCondition(
        name="工况 B：煤层锚杆",
        short_name="B",
        bolt_diameter_mm=22.0,
        yield_load_kn=205.0,
        hole_diameter_mm=30.0,
        bond_length_mm=1200.0,
        elastic_modulus_gpa=2.0,
        poisson_ratio=0.35,
        tray_side_mm=180.0,
        max_indent_mm=1.5,
        influence_depth_mm=900.0,
        torque_conversion=6.857,
        max_eccentricity_mm=30.0,
    ),
)


# ============================================================
# 2. 中文字体
# ============================================================

def setup_chinese_font() -> None:
    """自动选择本机可用的中文字体。"""

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "AR PL UMing CN",
        "Arial Unicode MS",
    ]

    for font_name in candidates:
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font_name]
            break
        except ValueError:
            continue

    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. 各约束计算
# ============================================================

def calculate_bond_limit_kn(condition: WorkingCondition) -> float:
    """锚固系统极限粘结力 F_bond，单位 kN。"""

    shear_strength_mpa = 0.25 * condition.elastic_modulus_gpa + 1.0
    return (
        math.pi
        * condition.hole_diameter_mm
        * condition.bond_length_mm
        * shear_strength_mpa
        * 1e-3
    )


def calculate_problem2_reference_torque_nm(condition: WorkingCondition) -> float:
    """
    问题二模型在不考虑偏心距时的最大允许预紧力矩。

    问题二只考虑：
    - 锚杆杆体屈服；
    - 锚固系统破坏；
    - 无偏心围岩压陷。

    不包含第三问新增的螺纹段复合应力约束。
    """

    bond_limit_kn = calculate_bond_limit_kn(condition)

    indentation_limit_e0_kn = (
        condition.max_indent_mm
        * condition.elastic_modulus_gpa
        * condition.tray_side_mm**2
        / (
            (1.0 - condition.poisson_ratio**2)
            * condition.influence_depth_mm
        )
    )

    problem2_preload_limit_kn = min(
        condition.yield_load_kn,
        bond_limit_kn,
        indentation_limit_e0_kn,
    )
    return condition.torque_conversion * problem2_preload_limit_kn


def calculate_constraint_curves(
    condition: WorkingCondition,
    eccentricity_mm: np.ndarray,
) -> dict[str, np.ndarray]:
    """计算第三问无钢带条件下的四类约束和综合下包络线。"""

    e = np.asarray(eccentricity_mm, dtype=float)
    d = condition.bolt_diameter_mm
    d_e = EFFECTIVE_DIAMETER_RATIO * d
    kd = condition.torque_conversion

    # 表中给出 Kd，因此无量纲扭矩系数 K=(Kd)/d
    torque_coefficient_k = kd / d

    # 1. 锚杆杆体屈服约束
    torque_yield_nm = np.full_like(
        e,
        kd * condition.yield_load_kn,
        dtype=float,
    )

    # 2. 锚固系统约束
    bond_limit_kn = calculate_bond_limit_kn(condition)
    torque_bond_nm = np.full_like(
        e,
        kd * bond_limit_kn,
        dtype=float,
    )

    # 3. 螺纹段屈服约束
    # 由拉伸、偏心弯曲和扭转共同作用下的 Von Mises 准则得到
    thread_yield_preload_kn = (
        math.pi
        * d_e**3
        * ALLOWABLE_STRESS_MPA
        * 1e-3
        / (
            4.0
            * np.sqrt(
                (d_e + 8.0 * e) ** 2
                + 48.0
                * THREAD_TORQUE_RATIO**2
                * torque_coefficient_k**2
                * d**2
            )
        )
    )
    torque_thread_yield_nm = kd * thread_yield_preload_kn

    # 4. 偏心修正后的围岩压陷约束
    indentation_preload_kn = (
        condition.max_indent_mm
        * condition.elastic_modulus_gpa
        * condition.tray_side_mm**2
        / (
            (1.0 + 6.0 * e / condition.tray_side_mm)
            * (1.0 - condition.poisson_ratio**2)
            * condition.influence_depth_mm
        )
    )
    torque_indent_nm = kd * indentation_preload_kn

    # 5. 综合最大允许预紧力矩
    torque_matrix = np.vstack(
        [
            torque_yield_nm,
            torque_bond_nm,
            torque_thread_yield_nm,
            torque_indent_nm,
        ]
    )
    torque_max_nm = np.min(torque_matrix, axis=0)

    return {
        "yield": torque_yield_nm,
        "bond": torque_bond_nm,
        "thread_yield": torque_thread_yield_nm,
        "indent": torque_indent_nm,
        "maximum": torque_max_nm,
    }


# ============================================================
# 4. 绘图
# ============================================================

def plot_condition(condition: WorkingCondition, output_path: Path) -> None:
    """生成单个工况的无钢带约束曲线图。"""

    full_contact_limit = condition.tray_side_mm / 6.0
    if condition.max_eccentricity_mm > full_contact_limit + 1e-9:
        raise ValueError(
            f"{condition.name} 的偏心距上限超过 b/6，"
            "线性全接触压强模型不再适用。"
        )

    point_count = int(round(condition.max_eccentricity_mm / ECCENTRIC_STEP_MM)) + 1
    eccentricity = np.linspace(
        0.0,
        condition.max_eccentricity_mm,
        point_count,
    )

    curves = calculate_constraint_curves(condition, eccentricity)

    # 这里的“不考虑偏心距”严格采用问题二模型，而不是令第三问综合模型 e=0
    problem2_reference_torque = calculate_problem2_reference_torque_nm(condition)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)

    # 先绘制综合下包络线作为较宽的背景带。
    # 之后再绘制各约束，使主控曲线与综合曲线重合时仍可看见。
    axis.plot(
        eccentricity,
        curves["maximum"],
        linewidth=5.0,
        alpha=0.35,
        zorder=1,
        label=r"综合最大允许预紧力矩 $T_{\max}(e)$",
    )

    axis.plot(
        eccentricity,
        curves["yield"],
        linewidth=1.5,
        zorder=2,
        label="锚杆杆体屈服约束",
    )
    axis.plot(
        eccentricity,
        curves["bond"],
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label="锚固系统约束",
    )
    axis.plot(
        eccentricity,
        curves["thread_yield"],
        linewidth=2.0,
        linestyle="-.",
        zorder=4,
        label="螺纹段屈服（复合应力）约束",
    )
    axis.plot(
        eccentricity,
        curves["indent"],
        linewidth=1.9,
        linestyle=":",
        zorder=3,
        label="围岩压陷约束",
    )

    # 问题二模型基准线
    axis.axhline(
        problem2_reference_torque,
        linewidth=1.7,
        linestyle=(0, (6, 3)),
        zorder=3,
        label=(
            "问题二无偏心模型："
            rf"$T_{{\max}}={problem2_reference_torque:.2f}\ \mathrm{{N\cdot m}}$"
        ),
    )

    # 在图内标注基准值
    axis.annotate(
        "问题二无偏心模型\n"
        rf"$T_{{\max}}={problem2_reference_torque:.2f}\ \mathrm{{N\cdot m}}$",
        xy=(condition.max_eccentricity_mm * 0.96, problem2_reference_torque),
        xytext=(
            condition.max_eccentricity_mm * 0.55,
            problem2_reference_torque * 1.06,
        ),
        arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        fontsize=9.5,
        ha="left",
        va="bottom",
    )

    axis.set_xlim(0.0, condition.max_eccentricity_mm)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("偏心距 $e$ / mm")
    axis.set_ylabel("最大允许预紧力矩 $T$ / N·m")
    axis.set_title(f"{condition.name}无钢带条件下各失效约束曲线")
    axis.grid(True, linestyle="--", alpha=0.35)

    # 图例顺序与原 question3_2.py 一致，并追加问题二基准线
    handles, labels = axis.get_legend_handles_labels()
    # 当前绘图顺序：
    # 0 综合，1 杆体屈服，2 锚固，3 螺纹段屈服，4 压陷，5 问题二基准
    legend_order = [1, 2, 3, 4, 0, 5]
    axis.legend(
        [handles[i] for i in legend_order],
        [labels[i] for i in legend_order],
        fontsize=8.8,
        loc="best",
    )

    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print("=" * 72)
    print(condition.name)
    print(f"偏心距范围：0 ≤ e ≤ {condition.max_eccentricity_mm:.1f} mm")
    print(f"完整接触上限：b/6 = {full_contact_limit:.1f} mm")
    print(
        "问题二无偏心模型最大允许预紧力矩："
        f"{problem2_reference_torque:.3f} N·m"
    )
    print(f"图片已保存：{output_path}")


# ============================================================
# 5. 主程序
# ============================================================

def main() -> None:
    setup_chinese_font()

    output_paths = {
        "A": OUTPUT_DIR / "condition_A_no_band_constraints.png",
        "B": OUTPUT_DIR / "condition_B_no_band_constraints.png",
    }

    for condition in CONDITIONS:
        plot_condition(condition, output_paths[condition.short_name])


if __name__ == "__main__":
    main()