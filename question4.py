import itertools
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FIGURE_DIR,
    TABLE_DIR,
    QUESTION4_CONSTRAINT_CURVES_FIGURE,
    QUESTION4_INTEGRITY_FACTOR,
    QUESTION4_MODULUS_RATIO,
    QUESTION4_MONTE_CARLO_FIGURE,
    QUESTION4_MONTE_CARLO_SAMPLES,
    QUESTION4_MONTE_CARLO_SUMMARY_TABLE,
    QUESTION4_MONTE_CARLO_WORST_TABLE,
    QUESTION4_MULTI_PERTURBATION_LEVELS,
    QUESTION4_RANDOM_SEED,
    QUESTION4_SAFETY_MARGIN_FACTOR,
    QUESTION4_SINGLE_PERTURBATIONS,
    QUESTION4_SINGLE_PERTURBATION_FIGURE,
    QUESTION4_SINGLE_PERTURBATION_TABLE,
    QUESTION4_VALIDITY_FIGURE,
    QUESTION4_VALIDITY_TABLE,
    QUESTION4_WORST_CASE_FIGURE,
    QUESTION4_WORST_CASE_LEVEL,
    QUESTION4_WORST_CASE_TABLE,
)
from utils import ensure_dir, setup_chinese_font


CONSTRAINT_NAMES = {
    "P1": "锚杆屈服",
    "P2": "锚固系统极限",
    "P3": "围岩压陷极限",
}

# 可靠性分析中直接扰动附录二原约束所涉及的独立参数。
# bolt_diameter 未单独进入当前三个原约束；Kd 已由 torque_conversion 表示。
PERTURBABLE_FIELDS = (
    "yield_load",
    "hole_diameter",
    "bond_length",
    "elastic_modulus",
    "poisson_ratio",
    "tray_side",
    "max_indent_depth",
    "influence_depth",
    "torque_conversion",
)

PARAMETER_LABELS = {
    "yield_load": "锚杆屈服载荷",
    "hole_diameter": "钻孔直径",
    "bond_length": "有效锚固长度",
    "elastic_modulus": "围岩弹性模量",
    "poisson_ratio": "泊松比",
    "tray_side": "托盘边长",
    "max_indent_depth": "允许压入深度",
    "influence_depth": "载荷影响深度",
    "torque_conversion": "力矩转换系数 Kd",
}


@dataclass(frozen=True)
class Question4Condition:
    """问题四单个验证工况的参数。"""

    name: str
    protodyakonov_f: float        # 普氏系数 f
    modulus_ratio: float          # 模量比 M_R
    integrity_factor: float       # 完整性折减系数 eta

    bolt_diameter: float          # 锚杆直径 d，mm（保留用于说明）
    yield_load: float             # P_yield，kN
    hole_diameter: float          # D_hole，mm
    bond_length: float            # L_bond，mm
    elastic_modulus: float        # 表4/原约束使用的 E，GPa
    poisson_ratio: float          # nu
    tray_side: float              # b，mm
    max_indent_depth: float       # delta_max，mm
    influence_depth: float        # h0，mm
    torque_conversion: float      # Kd，N·m/kN


@dataclass(frozen=True)
class LimitResult:
    """三个预紧力上限及其附加信息。"""

    p1: float
    p2: float
    p3: float
    elastic_modulus: float
    shear_strength: float

    @property
    def limits(self) -> dict[str, float]:
        return {"P1": self.p1, "P2": self.p2, "P3": self.p3}

    @property
    def minimum_preload(self) -> float:
        return min(self.p1, self.p2, self.p3)

    @property
    def controlling_constraint(self) -> str:
        return min(self.limits, key=self.limits.get)


@dataclass(frozen=True)
class ReliabilityEvaluation:
    """固定施工力矩在某组实际参数下的安全性评价。"""

    actual_preload: float
    p1: float
    p2: float
    p3: float
    controlling_constraint: str
    minimum_safety_factor: float
    minimum_margin_kn: float
    maximum_allowable_torque: float
    is_safe: bool


# ---------------------------------------------------------------------
# 基础模型与原始约束
# ---------------------------------------------------------------------


def elastic_modulus_from_f(
    protodyakonov_f: float,
    modulus_ratio: float,
    integrity_factor: float,
) -> float:
    """由 E = 0.01 * eta * M_R * f 计算围岩弹性模量，单位 GPa。"""
    return 0.01 * integrity_factor * modulus_ratio * protodyakonov_f


def infer_f_from_elastic_modulus(
    elastic_modulus: float,
    modulus_ratio: float,
    integrity_factor: float,
) -> float:
    """由表4弹性模量反推与基准 M_R、eta 一致的普氏系数。"""
    denominator = 0.01 * integrity_factor * modulus_ratio
    if denominator <= 0:
        raise ValueError("模量比与完整性折减系数必须为正数。")
    return elastic_modulus / denominator


def calculate_question4_model_limits(
    condition: Question4Condition,
) -> LimitResult:
    """按论文式（44）—（48）计算 P1、P2、P3。"""
    e_model = elastic_modulus_from_f(
        condition.protodyakonov_f,
        condition.modulus_ratio,
        condition.integrity_factor,
    )

    # E = 0.01 * eta * M_R * f (GPa)，故
    # tau = 0.25E + 1 = 0.25 * eta * M_R * f * 1e-2 + 1。
    shear_strength = 0.25 * e_model + 1.0

    p1 = condition.yield_load
    p2 = (
        math.pi
        * condition.hole_diameter
        * condition.bond_length
        * shear_strength
        * 1e-3
    )
    p3 = (
        condition.max_indent_depth
        * e_model
        * condition.tray_side**2
        / (
            (1.0 - condition.poisson_ratio**2)
            * condition.influence_depth
        )
    )

    return LimitResult(
        p1=float(p1),
        p2=float(p2),
        p3=float(p3),
        elastic_modulus=float(e_model),
        shear_strength=float(shear_strength),
    )


def calculate_original_limits(condition: Question4Condition) -> LimitResult:
    """按附录二原公式、直接使用实际参数 E 计算三个约束上限。"""
    if not 0 <= condition.poisson_ratio < 1:
        raise ValueError("泊松比必须满足 0 <= nu < 1。")

    shear_strength = 0.25 * condition.elastic_modulus + 1.0
    p1 = condition.yield_load
    p2 = (
        math.pi
        * condition.hole_diameter
        * condition.bond_length
        * shear_strength
        * 1e-3
    )
    p3 = (
        condition.max_indent_depth
        * condition.elastic_modulus
        * condition.tray_side**2
        / (
            (1.0 - condition.poisson_ratio**2)
            * condition.influence_depth
        )
    )

    return LimitResult(
        p1=float(p1),
        p2=float(p2),
        p3=float(p3),
        elastic_modulus=float(condition.elastic_modulus),
        shear_strength=float(shear_strength),
    )


def calculate_nominal_topt(
    condition: Question4Condition,
    safety_margin_factor: float = QUESTION4_SAFETY_MARGIN_FACTOR,
) -> dict[str, float | str]:
    """按论文式（49）计算 T_opt = gamma * Kd * min(P1, P2, P3)。"""
    if not 0.0 < safety_margin_factor <= 1.0:
        raise ValueError("安全裕度系数 gamma 必须满足 0 < gamma <= 1。")

    model_limits = calculate_question4_model_limits(condition)
    boundary_preload = model_limits.minimum_preload
    boundary_torque = condition.torque_conversion * boundary_preload
    optimal_preload = safety_margin_factor * boundary_preload
    optimal_torque = safety_margin_factor * boundary_torque

    return {
        "safety_margin_factor": float(safety_margin_factor),
        "boundary_preload_kn": float(boundary_preload),
        "boundary_torque_nm": float(boundary_torque),
        "optimal_preload_kn": float(optimal_preload),
        "optimal_torque_nm": float(optimal_torque),
        "controlling_constraint": model_limits.controlling_constraint,
        "model_elastic_modulus_gpa": model_limits.elastic_modulus,
        "model_shear_strength_mpa": model_limits.shear_strength,
    }


def evaluate_fixed_torque(
    condition: Question4Condition,
    fixed_torque_nm: float,
) -> ReliabilityEvaluation:
    """将固定施工力矩换算为实际预紧力，并代入附录二原约束检验。"""
    if condition.torque_conversion <= 0:
        raise ValueError("力矩转换系数 Kd 必须为正数。")

    original = calculate_original_limits(condition)
    actual_preload = fixed_torque_nm / condition.torque_conversion
    ratios = {
        key: value / actual_preload
        for key, value in original.limits.items()
    }
    controlling = min(original.limits, key=original.limits.get)
    minimum_limit = original.limits[controlling]
    minimum_safety_factor = min(ratios.values())
    minimum_margin = minimum_limit - actual_preload
    maximum_allowable_torque = condition.torque_conversion * minimum_limit

    return ReliabilityEvaluation(
        actual_preload=float(actual_preload),
        p1=original.p1,
        p2=original.p2,
        p3=original.p3,
        controlling_constraint=controlling,
        minimum_safety_factor=float(minimum_safety_factor),
        minimum_margin_kn=float(minimum_margin),
        maximum_allowable_torque=float(maximum_allowable_torque),
        is_safe=bool(minimum_safety_factor >= 1.0 - 1e-12),
    )


# ---------------------------------------------------------------------
# 有效性验证
# ---------------------------------------------------------------------


def build_validity_table(
    conditions: list[Question4Condition],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成有效性逐约束表和工况汇总表。"""
    detail_records = []
    summary_records = []

    for condition in conditions:
        nominal = calculate_nominal_topt(condition)
        fixed_torque = float(nominal["optimal_torque_nm"])
        evaluation = evaluate_fixed_torque(condition, fixed_torque)
        original = calculate_original_limits(condition)

        for key, limit in original.limits.items():
            torque_limit = condition.torque_conversion * limit
            detail_records.append(
                {
                    "工况": condition.name,
                    "普氏系数_f": condition.protodyakonov_f,
                    "约束": key,
                    "失效模式": CONSTRAINT_NAMES[key],
                    "原约束预紧力上限_kN": limit,
                    "原约束力矩上限_Nm": torque_limit,
                    "安全裕度系数_gamma": nominal["safety_margin_factor"],
                    "模型边界力矩_Nm": nominal["boundary_torque_nm"],
                    "模型Topt_Nm": fixed_torque,
                    "Topt对应实际预紧力_kN": evaluation.actual_preload,
                    "预紧力安全系数": limit / evaluation.actual_preload,
                    "力矩裕度_Nm": torque_limit - fixed_torque,
                    "是否满足": fixed_torque <= torque_limit + 1e-9,
                }
            )

        summary_records.append(
            {
                "工况": condition.name,
                "普氏系数_f": condition.protodyakonov_f,
                "表4弹性模量_GPa": condition.elastic_modulus,
                "模型弹性模量_GPa": nominal["model_elastic_modulus_gpa"],
                "模型主控约束": nominal["controlling_constraint"],
                "原约束主控约束": evaluation.controlling_constraint,
                "安全裕度系数_gamma": nominal["safety_margin_factor"],
                "模型边界力矩_Nm": nominal["boundary_torque_nm"],
                "Topt_Nm": fixed_torque,
                "对应实际预紧力_kN": evaluation.actual_preload,
                "最小安全系数": evaluation.minimum_safety_factor,
                "最小预紧力裕度_kN": evaluation.minimum_margin_kn,
                "有效性是否通过": evaluation.is_safe,
            }
        )

    return pd.DataFrame(detail_records), pd.DataFrame(summary_records)


# ---------------------------------------------------------------------
# 参数扰动工具
# ---------------------------------------------------------------------


def _bounded_parameter_value(field_name: str, value: float) -> float:
    """确保扰动后参数仍处于基本物理可行范围。"""
    if field_name == "poisson_ratio":
        return float(np.clip(value, 0.01, 0.49))
    return max(float(value), 1e-12)


def perturb_condition(
    condition: Question4Condition,
    perturbations: dict[str, float],
) -> Question4Condition:
    """按相对变化率扰动工况参数。"""
    updates = {}
    for field_name, relative_change in perturbations.items():
        base_value = float(getattr(condition, field_name))
        new_value = base_value * (1.0 + relative_change)
        updates[field_name] = _bounded_parameter_value(field_name, new_value)
    return replace(condition, **updates)


def evaluation_to_record(
    condition_name: str,
    evaluation: ReliabilityEvaluation,
) -> dict[str, float | str | bool]:
    return {
        "工况": condition_name,
        "实际预紧力_kN": evaluation.actual_preload,
        "P1_锚杆屈服上限_kN": evaluation.p1,
        "P2_锚固上限_kN": evaluation.p2,
        "P3_压陷上限_kN": evaluation.p3,
        "主控约束": evaluation.controlling_constraint,
        "主控失效模式": CONSTRAINT_NAMES[evaluation.controlling_constraint],
        "最小安全系数": evaluation.minimum_safety_factor,
        "最小裕度_kN": evaluation.minimum_margin_kn,
        "最大允许力矩_Nm": evaluation.maximum_allowable_torque,
        "是否安全": evaluation.is_safe,
    }


# ---------------------------------------------------------------------
# 单参数扰动
# ---------------------------------------------------------------------


def run_single_parameter_perturbation(
    conditions: list[Question4Condition],
    perturbation_levels: tuple[float, ...] = QUESTION4_SINGLE_PERTURBATIONS,
) -> pd.DataFrame:
    """保持含安全裕度的名义 T_opt 不变，逐一扰动一个参数并重新检验原约束。"""
    records = []

    for condition in conditions:
        fixed_torque = float(calculate_nominal_topt(condition)["optimal_torque_nm"])

        # 加入零扰动基线，便于作图。
        levels_with_baseline = tuple(sorted(set((*perturbation_levels, 0.0))))
        for field_name in PERTURBABLE_FIELDS:
            for level in levels_with_baseline:
                perturbed = perturb_condition(condition, {field_name: level})
                evaluation = evaluate_fixed_torque(perturbed, fixed_torque)
                records.append(
                    {
                        **evaluation_to_record(condition.name, evaluation),
                        "参数": field_name,
                        "参数名称": PARAMETER_LABELS[field_name],
                        "扰动比例": level,
                        "扰动百分比": 100.0 * level,
                        "扰动后参数值": getattr(perturbed, field_name),
                        "固定Topt_Nm": fixed_torque,
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# 多参数随机扰动（Monte Carlo）
# ---------------------------------------------------------------------


def run_multi_parameter_monte_carlo(
    conditions: list[Question4Condition],
    levels: tuple[float, ...] = QUESTION4_MULTI_PERTURBATION_LEVELS,
    samples: int = QUESTION4_MONTE_CARLO_SAMPLES,
    seed: int = QUESTION4_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """所有原约束参数同时独立均匀扰动，统计固定含裕度 T_opt 的安全概率。"""
    rng = np.random.default_rng(seed)
    raw_records = []
    summary_records = []

    for condition_index, condition in enumerate(conditions):
        fixed_torque = float(calculate_nominal_topt(condition)["optimal_torque_nm"])

        for level_index, amplitude in enumerate(levels):
            condition_records = []
            # 每个工况、每个幅度使用稳定但不同的随机流。
            local_rng = np.random.default_rng(
                rng.integers(0, np.iinfo(np.int32).max)
                + condition_index * 100
                + level_index
            )

            for sample_index in range(samples):
                perturbations = {
                    field_name: float(local_rng.uniform(-amplitude, amplitude))
                    for field_name in PERTURBABLE_FIELDS
                }
                perturbed = perturb_condition(condition, perturbations)
                evaluation = evaluate_fixed_torque(perturbed, fixed_torque)
                record = {
                    **evaluation_to_record(condition.name, evaluation),
                    "扰动幅度": amplitude,
                    "扰动幅度百分比": 100.0 * amplitude,
                    "样本编号": sample_index,
                    "固定Topt_Nm": fixed_torque,
                }
                for field_name, change in perturbations.items():
                    record[f"扰动_{field_name}"] = change
                condition_records.append(record)

            raw_records.extend(condition_records)
            subset = pd.DataFrame(condition_records)
            summary_records.append(
                {
                    "工况": condition.name,
                    "扰动幅度": amplitude,
                    "扰动幅度百分比": 100.0 * amplitude,
                    "样本数": len(subset),
                    "安全样本数": int(subset["是否安全"].sum()),
                    "安全率": float(subset["是否安全"].mean()),
                    "最小安全系数": float(subset["最小安全系数"].min()),
                    "5%分位安全系数": float(subset["最小安全系数"].quantile(0.05)),
                    "中位安全系数": float(subset["最小安全系数"].median()),
                    "平均安全系数": float(subset["最小安全系数"].mean()),
                    "最小裕度_kN": float(subset["最小裕度_kN"].min()),
                }
            )

    return pd.DataFrame(raw_records), pd.DataFrame(summary_records)


# ---------------------------------------------------------------------
# 最坏情况：枚举扰动区间所有顶点
# ---------------------------------------------------------------------


def run_worst_case_corner_search(
    conditions: list[Question4Condition],
    amplitude: float = QUESTION4_WORST_CASE_LEVEL,
    safety_margin_factor: float = QUESTION4_SAFETY_MARGIN_FACTOR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    枚举所有参数在 ±amplitude 处的组合，寻找固定含裕度 T_opt 的最坏情况。

    当前有 9 个扰动参数，共 2^9=512 个角点，计算量很小。
    """
    worst_records = []
    all_corner_records = []

    signs = (-amplitude, amplitude)
    for condition in conditions:
        nominal_torque = float(calculate_nominal_topt(condition)["optimal_torque_nm"])
        condition_records = []

        for corner_index, changes in enumerate(
            itertools.product(signs, repeat=len(PERTURBABLE_FIELDS))
        ):
            perturbations = dict(zip(PERTURBABLE_FIELDS, changes))
            perturbed = perturb_condition(condition, perturbations)
            evaluation = evaluate_fixed_torque(perturbed, nominal_torque)
            record = {
                **evaluation_to_record(condition.name, evaluation),
                "角点编号": corner_index,
                "固定名义Topt_Nm": nominal_torque,
            }
            for field_name, change in perturbations.items():
                record[f"扰动_{field_name}"] = change
            condition_records.append(record)

        corners = pd.DataFrame(condition_records)
        all_corner_records.extend(condition_records)
        worst_row = corners.loc[corners["最小安全系数"].idxmin()].copy()

        # 最坏角点下的最大允许力矩即该不确定区间内的稳健上界。
        worst_allowable_torque = float(worst_row["最大允许力矩_Nm"])
        robust_recommended_torque = safety_margin_factor * worst_allowable_torque

        # 用推荐稳健力矩复核全部角点。
        robust_factors = []
        for _, corner_row in corners.iterrows():
            perturbations = {
                field_name: float(corner_row[f"扰动_{field_name}"])
                for field_name in PERTURBABLE_FIELDS
            }
            perturbed = perturb_condition(condition, perturbations)
            robust_eval = evaluate_fixed_torque(
                perturbed,
                robust_recommended_torque,
            )
            robust_factors.append(robust_eval.minimum_safety_factor)

        worst_record = dict(worst_row)
        worst_record.update(
            {
                "扰动幅度": amplitude,
                "扰动幅度百分比": 100.0 * amplitude,
                "名义Topt绝对稳健": bool(corners["是否安全"].all()),
                "角点安全率": float(corners["是否安全"].mean()),
                "最坏允许力矩_Nm": worst_allowable_torque,
                "名义Topt相对最坏上限比": nominal_torque / worst_allowable_torque,
                "稳健建议安全裕度系数_gamma": safety_margin_factor,
                "稳健建议力矩_Nm": robust_recommended_torque,
                "稳健建议力矩相对名义值": robust_recommended_torque / nominal_torque,
                "稳健建议力矩角点最小安全系数": min(robust_factors),
                "稳健建议力矩全部角点安全": min(robust_factors) >= 1.0 - 1e-12,
            }
        )
        worst_records.append(worst_record)

    return pd.DataFrame(worst_records), pd.DataFrame(all_corner_records)


# ---------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------


def plot_validity(
    validity_detail: pd.DataFrame,
    output_path: Path = QUESTION4_VALIDITY_FIGURE,
) -> None:
    """绘制各工况实际预紧力与三个原约束上限的比较。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)

    conditions = list(validity_detail["工况"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(7 * len(conditions), 5),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, condition_name in zip(axes[0], conditions):
        subset = validity_detail[validity_detail["工况"] == condition_name]
        labels = list(subset["失效模式"])
        values = list(subset["原约束预紧力上限_kN"])
        actual_preload = float(subset["Topt对应实际预紧力_kN"].iloc[0])

        x = np.arange(len(labels))
        axis.bar(x, values, label="原约束上限")
        axis.axhline(actual_preload, linestyle="--", linewidth=1.5, label="Topt 对应预紧力")
        axis.set_xticks(x, labels, rotation=15)
        axis.set_ylabel("预紧力 / kN")
        axis.set_title(condition_name)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        axis.legend(fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_single_parameter_sensitivity(
    data: pd.DataFrame,
    output_path: Path = QUESTION4_SINGLE_PERTURBATION_FIGURE,
) -> None:
    """绘制单参数扰动下的最小安全系数曲线。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)

    conditions = list(data["工况"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(8 * len(conditions), 5.5),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, condition_name in zip(axes[0], conditions):
        subset = data[data["工况"] == condition_name]
        for parameter_name, group in subset.groupby("参数名称"):
            group = group.sort_values("扰动百分比")
            axis.plot(
                group["扰动百分比"],
                group["最小安全系数"],
                marker="o",
                linewidth=1.1,
                label=parameter_name,
            )

        axis.axhline(1.0, linestyle="--", linewidth=1.5, label="安全临界线")
        axis.set_xlabel("单参数扰动 / %")
        axis.set_ylabel("最小安全系数")
        axis.set_title(condition_name)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=8, ncol=2)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_monte_carlo(
    raw_data: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path = QUESTION4_MONTE_CARLO_FIGURE,
) -> None:
    """绘制多参数随机扰动的最小安全系数分布。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)

    groups = list(
        raw_data[["工况", "扰动幅度百分比"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(9, 4.2 * len(groups)),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, (condition_name, amplitude_pct) in zip(axes[:, 0], groups):
        subset = raw_data[
            (raw_data["工况"] == condition_name)
            & np.isclose(raw_data["扰动幅度百分比"], amplitude_pct)
        ]
        summary_row = summary[
            (summary["工况"] == condition_name)
            & np.isclose(summary["扰动幅度百分比"], amplitude_pct)
        ].iloc[0]

        axis.hist(subset["最小安全系数"], bins=45, alpha=0.85)
        axis.axvline(1.0, linestyle="--", linewidth=1.5, label="安全临界线")
        axis.set_xlabel("最小安全系数")
        axis.set_ylabel("样本数")
        axis.set_title(
            f"{condition_name}：多参数 ±{amplitude_pct:.0f}% 扰动，"
            f"安全率={summary_row['安全率']:.2%}"
        )
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        axis.legend(fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_worst_case(
    worst_case: pd.DataFrame,
    output_path: Path = QUESTION4_WORST_CASE_FIGURE,
) -> None:
    """比较名义 Topt、最坏允许力矩和稳健建议力矩。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)

    labels = list(worst_case["工况"])
    x = np.arange(len(labels))
    width = 0.25

    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    axis.bar(
        x - width,
        worst_case["固定名义Topt_Nm"],
        width,
        label="名义 Topt",
    )
    axis.bar(
        x,
        worst_case["最坏允许力矩_Nm"],
        width,
        label="±20%最坏允许力矩",
    )
    axis.bar(
        x + width,
        worst_case["稳健建议力矩_Nm"],
        width,
        label="稳健建议力矩",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("力矩 / N·m")
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    axis.legend(fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_constraint_curves(
    conditions: list[Question4Condition],
    output_path: Path = QUESTION4_CONSTRAINT_CURVES_FIGURE,
    f_min: float = 0.2,
    f_max: float = 10.0,
) -> None:
    """绘制 P1、P2、P3 与 f 的关系及综合上限。"""
    setup_chinese_font()
    ensure_dir(output_path.parent)

    f_values = np.linspace(f_min, f_max, 400)
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(7 * len(conditions), 5.2),
        constrained_layout=True,
        squeeze=False,
    )

    for axis, condition in zip(axes[0], conditions):
        p1_values = []
        p2_values = []
        p3_values = []
        pmin_values = []
        popt_values = []

        for f_value in f_values:
            current = replace(
                condition,
                protodyakonov_f=float(f_value),
                elastic_modulus=elastic_modulus_from_f(
                    float(f_value),
                    condition.modulus_ratio,
                    condition.integrity_factor,
                ),
            )
            limits = calculate_question4_model_limits(current)
            p1_values.append(limits.p1)
            p2_values.append(limits.p2)
            p3_values.append(limits.p3)
            pmin_values.append(limits.minimum_preload)
            popt_values.append(QUESTION4_SAFETY_MARGIN_FACTOR * limits.minimum_preload)

        axis.plot(f_values, p1_values, label="P1 锚杆屈服")
        axis.plot(f_values, p2_values, label="P2 锚固系统")
        axis.plot(f_values, p3_values, label="P3 围岩压陷")
        axis.plot(f_values, pmin_values, linewidth=2.0, label="失效边界 min(P1,P2,P3)")
        axis.plot(
            f_values,
            popt_values,
            linewidth=2.2,
            label=f"推荐预紧力 gamma·min(P1,P2,P3)，gamma={QUESTION4_SAFETY_MARGIN_FACTOR:.2f}",
        )
        axis.axvline(3.0, linestyle="--", linewidth=1.0)
        axis.axvline(6.0, linestyle="--", linewidth=1.0)
        axis.set_xlabel("普氏系数 f")
        axis.set_ylabel("允许预紧力 / kN")
        axis.set_title(condition.name)
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=8)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# 默认工况、输出与主程序
# ---------------------------------------------------------------------


def build_default_conditions() -> list[Question4Condition]:
    """
    使用问题二表4工况参数。

    表4未直接给出 f，故在 M_R=300、eta=1 的基准下，
    由表4弹性模量 E 反推 f，使问题四模型得到的 E 与原参数一致。
    """
    mr = QUESTION4_MODULUS_RATIO
    eta = QUESTION4_INTEGRITY_FACTOR

    e_a = 15.0
    e_b = 2.0

    condition_a = Question4Condition(
        name="工况 A：岩层锚杆",
        protodyakonov_f=infer_f_from_elastic_modulus(e_a, mr, eta),
        modulus_ratio=mr,
        integrity_factor=eta,
        bolt_diameter=20.0,
        yield_load=170.0,
        hole_diameter=28.0,
        bond_length=800.0,
        elastic_modulus=e_a,
        poisson_ratio=0.25,
        tray_side=150.0,
        max_indent_depth=1.0,
        influence_depth=900.0,
        torque_conversion=4.272,
    )

    condition_b = Question4Condition(
        name="工况 B：煤层锚杆",
        protodyakonov_f=infer_f_from_elastic_modulus(e_b, mr, eta),
        modulus_ratio=mr,
        integrity_factor=eta,
        bolt_diameter=22.0,
        yield_load=205.0,
        hole_diameter=30.0,
        bond_length=1200.0,
        elastic_modulus=e_b,
        poisson_ratio=0.35,
        tray_side=180.0,
        max_indent_depth=1.5,
        influence_depth=900.0,
        # 按论文中修正后的 K=0.3117 取 Kd=0.3117*22≈6.8574。
        # 当前 question2.py 中的 6.611 对应未修正的 K=0.3005，需注意统一。
        torque_conversion=6.8574,
    )

    return [condition_a, condition_b]


def save_tables(
    validity_detail: pd.DataFrame,
    single_data: pd.DataFrame,
    monte_carlo_raw: pd.DataFrame,
    monte_carlo_summary: pd.DataFrame,
    worst_case: pd.DataFrame,
) -> None:
    """保存论文可直接整理使用的 CSV 表格。"""
    ensure_dir(TABLE_DIR)
    validity_detail.to_csv(QUESTION4_VALIDITY_TABLE, index=False, encoding="utf-8-sig")
    single_data.to_csv(QUESTION4_SINGLE_PERTURBATION_TABLE, index=False, encoding="utf-8-sig")
    monte_carlo_summary.to_csv(
        QUESTION4_MONTE_CARLO_SUMMARY_TABLE,
        index=False,
        encoding="utf-8-sig",
    )

    # 只保存每组最不利的 50 个随机样本，避免结果文件过大。
    worst_random = (
        monte_carlo_raw
        .sort_values(["工况", "扰动幅度百分比", "最小安全系数"])
        .groupby(["工况", "扰动幅度百分比"], as_index=False, group_keys=False)
        .head(50)
    )
    worst_random.to_csv(
        QUESTION4_MONTE_CARLO_WORST_TABLE,
        index=False,
        encoding="utf-8-sig",
    )
    worst_case.to_csv(QUESTION4_WORST_CASE_TABLE, index=False, encoding="utf-8-sig")


def print_summary(
    validity_summary: pd.DataFrame,
    monte_carlo_summary: pd.DataFrame,
    worst_case: pd.DataFrame,
) -> None:
    """输出主要验证结论。"""
    print("=" * 72)
    print("问题四：有效性验证")
    print("=" * 72)
    for _, row in validity_summary.iterrows():
        print(
            f"{row['工况']}：f={row['普氏系数_f']:.4f}, "
            f"gamma={row['安全裕度系数_gamma']:.2f}, "
            f"边界力矩={row['模型边界力矩_Nm']:.3f} N·m, "
            f"Topt={row['Topt_Nm']:.3f} N·m, "
            f"最小安全系数={row['最小安全系数']:.4f}, "
            f"有效性={'通过' if row['有效性是否通过'] else '未通过'}"
        )

    print("\n" + "=" * 72)
    print("问题四：多参数随机扰动可靠性")
    print("=" * 72)
    for _, row in monte_carlo_summary.iterrows():
        print(
            f"{row['工况']}，±{row['扰动幅度百分比']:.0f}%："
            f"安全率={row['安全率']:.2%}, "
            f"最小安全系数={row['最小安全系数']:.4f}, "
            f"5%分位={row['5%分位安全系数']:.4f}"
        )

    print("\n" + "=" * 72)
    print("问题四：±20%角点最坏情况")
    print("=" * 72)
    for _, row in worst_case.iterrows():
        print(
            f"{row['工况']}：名义Topt={row['固定名义Topt_Nm']:.3f} N·m, "
            f"最坏允许力矩={row['最坏允许力矩_Nm']:.3f} N·m, "
            f"绝对稳健={'是' if row['名义Topt绝对稳健'] else '否'}, "
            f"稳健建议力矩={row['稳健建议力矩_Nm']:.3f} N·m"
        )


def main() -> None:
    ensure_dir(FIGURE_DIR)
    ensure_dir(TABLE_DIR)
    conditions = build_default_conditions()

    # 1. 有效性验证
    validity_detail, validity_summary = build_validity_table(conditions)

    # 2. 单参数扰动
    single_data = run_single_parameter_perturbation(conditions)

    # 3. 多参数随机扰动
    monte_carlo_raw, monte_carlo_summary = run_multi_parameter_monte_carlo(conditions)

    # 4. ±20% 参数区间角点最坏情况
    worst_case, _ = run_worst_case_corner_search(conditions)

    # 5. 保存图表
    plot_validity(validity_detail)
    plot_single_parameter_sensitivity(single_data)
    plot_monte_carlo(monte_carlo_raw, monte_carlo_summary)
    plot_worst_case(worst_case)
    plot_constraint_curves(conditions)
    save_tables(
        validity_detail,
        single_data,
        monte_carlo_raw,
        monte_carlo_summary,
        worst_case,
    )

    print_summary(validity_summary, monte_carlo_summary, worst_case)


if __name__ == "__main__":
    main()
