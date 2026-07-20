import math
from dataclasses import dataclass


@dataclass
class WorkingCondition:
    """单个工况的地质与支护参数。"""
    name: str
    bolt_diameter: float          # 锚杆直径 d，mm
    yield_load: float             # 锚杆屈服载荷 P_yield，kN
    hole_diameter: float          # 钻孔直径 D_hole，mm
    bond_length: float            # 有效锚固长度 L_bond，mm
    elastic_modulus: float        # 围岩弹性模量 E，GPa
    poisson_ratio: float          # 泊松比 ν
    tray_side: float              # 托盘边长 b，mm
    max_indent_depth: float       # 允许压入深度 δ_max，mm
    influence_depth: float        # 围岩载荷影响深度 h0，mm

    # 这里填写表格中的 4.272、6.01
    # 根据数值大小判断，它们更可能是拟合得到的 Kd，而不是无量纲 K
    torque_conversion: float      # Kd，单位 N·m/kN


def calculate_condition(condition: WorkingCondition) -> dict:
    """计算单个工况的最大预紧力矩、预紧力及钢带必要性。"""

    # 1. 围岩与锚固剂界面的剪切强度，MPa
    shear_strength = 0.25 * condition.elastic_modulus + 1.0

    # 2. 锚固系统极限粘结力，kN
    bond_limit = (
        math.pi
        * condition.hole_diameter
        * condition.bond_length
        * shear_strength
        * 1e-3
    )

    # 3. 仅使用托盘时，围岩压陷对应的预紧力上限，kN
    indentation_limit = (
        condition.max_indent_depth
        * condition.elastic_modulus
        * condition.tray_side ** 2
        / (
            (1 - condition.poisson_ratio ** 2)
            * condition.influence_depth
        )
    )

    # 4. 不考虑围岩压陷时，锚杆和锚固系统的结构承载上限，kN
    structural_limit = min(
        condition.yield_load,
        bond_limit
    )

    # 5. 钢带必要性判断
    need_steel_band = indentation_limit < structural_limit

    # 6. 不使用钢带时的最大允许预紧力，kN
    max_preload_without_band = min(
        condition.yield_load,
        bond_limit,
        indentation_limit
    )

    # 7. 允许在必要时使用钢带时的最大允许预紧力，kN
    # 使用钢带后，围岩压陷不再作为当前模型中的主控上限
    max_preload_with_band = structural_limit

    # 8. 最大预紧力矩
    # 表中的 4.272、6.01 按 Kd 处理，因此 T = (Kd)P
    max_torque_without_band = (
        condition.torque_conversion
        * max_preload_without_band
    )

    max_torque_with_band = (
        condition.torque_conversion
        * max_preload_with_band
    )

    return {
        "工况": condition.name,
        "剪切强度_tau_MPa": shear_strength,
        "锚杆屈服上限_kN": condition.yield_load,
        "锚固极限_kN": bond_limit,
        "压陷极限_kN": indentation_limit,
        "无钢带最大预紧力_kN": max_preload_without_band,
        "允许使用钢带时最大预紧力_kN": max_preload_with_band,
        "无钢带最大预紧力矩_Nm": max_torque_without_band,
        "允许使用钢带时最大预紧力矩_Nm": max_torque_with_band,
        "是否需要钢带": need_steel_band,
    }


def print_result(result: dict) -> None:
    """按论文计算结果的形式输出。"""

    print("=" * 60)
    print(result["工况"])
    print("-" * 60)

    print(f"界面剪切强度 τ："
          f"{result['剪切强度_tau_MPa']:.3f} MPa")

    print(f"锚杆屈服载荷："
          f"{result['锚杆屈服上限_kN']:.3f} kN")

    print(f"锚固系统极限粘结力："
          f"{result['锚固极限_kN']:.3f} kN")

    print(f"围岩压陷对应预紧力上限："
          f"{result['压陷极限_kN']:.3f} kN")

    if result["是否需要钢带"]:
        print("钢带判断：必须使用钢带")
        print(
            "原因：围岩压陷上限小于锚杆—锚固系统的结构承载上限。"
        )
    else:
        print("钢带判断：无需使用钢带")
        print(
            "原因：围岩压陷上限不小于锚杆—锚固系统的结构承载上限。"
        )

    print(f"无钢带时最大预紧力："
          f"{result['无钢带最大预紧力_kN']:.3f} kN")

    print(f"无钢带时最大预紧力矩："
          f"{result['无钢带最大预紧力矩_Nm']:.3f} N·m")

    print(f"允许按需使用钢带时最大预紧力："
          f"{result['允许使用钢带时最大预紧力_kN']:.3f} kN")

    print(f"允许按需使用钢带时最大预紧力矩："
          f"{result['允许使用钢带时最大预紧力矩_Nm']:.3f} N·m")


def main() -> None:
    condition_a = WorkingCondition(
        name="工况 A：岩层锚杆",
        bolt_diameter=20,
        yield_load=170,
        hole_diameter=28,
        bond_length=800,
        elastic_modulus=15,
        poisson_ratio=0.25,
        tray_side=150,
        max_indent_depth=1.0,
        influence_depth=900,
        torque_conversion=4.272
    )

    condition_b = WorkingCondition(
        name="工况 B：煤层锚杆",
        bolt_diameter=22,
        yield_load=205,
        hole_diameter=30,
        bond_length=1200,
        elastic_modulus=2,
        poisson_ratio=0.35,
        tray_side=180,
        max_indent_depth=1.5,
        influence_depth=900,
        torque_conversion=6.857
    )

    for condition in (condition_a, condition_b):
        result = calculate_condition(condition)
        print_result(result)


if __name__ == "__main__":
    main()