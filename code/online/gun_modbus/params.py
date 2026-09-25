"""与《电子枪项目需求》一致的 5 路电参及 Modbus 读写模式名。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# 写保持寄存器用的 mode（与旧 project/modbus.py 一致）
SET_FILAMENT: Final = "FILAMENT"
SET_CURRENT: Final = "CURRENT"
SET_VOLTAGE: Final = "VOLTAGE"
SET_BIAS: Final = "BIAS"
SET_N2K: Final = "N2K"

# 读输入寄存器反馈用的 mode
FBK_HV: Final = "HV FBK"
FBK_MA: Final = "mA FBK"
FBK_N2K_V: Final = "N2K VFBK"
FBK_BIAS_V: Final = "BIAS VFBK"
FBK_FIL_I: Final = "FIL IFBK"

# 使能寄存器地址（set_state_on_or_off）
STATE_HV: Final = 1
STATE_BIAS: Final = 2
STATE_BEAM: Final = 3
STATE_FIL: Final = 4
STATE_REMOTE: Final = 10
STATE_N2K: Final = 11

# CLI / JSON 未指定栅偏时的默认值（与 param_limits.bias_v 固定值一致）
DEFAULT_BIAS_V: Final = 200.0


@dataclass
class GunParams:
    """PDF 五参：ANODE V/I、CATHODE V、BIAS V、FIL I。"""

    anode_kv: float
    anode_ua: float
    cathode_v: float
    bias_v: float
    fil_a: float


@dataclass(frozen=True)
class GunParamLimits:
    """五参允许范围（闭区间），默认与 gun_modbus_config.json 一致。"""

    anode_kv_min: float = 70.0
    anode_kv_max: float = 110.0
    anode_ua_min: float = 30.0
    anode_ua_max: float = 300.0
    cathode_v_min: float = 500.0
    cathode_v_max: float = 750.0
    bias_v_min: float = 200.0
    bias_v_max: float = 200.0
    fil_a_min: float = 0.42
    fil_a_max: float = 0.46


class GunParamsOutOfRangeError(ValueError):
    """五参超出 ``GunParamLimits`` 时抛出。"""


def validate_gun_params(params: GunParams, limits: GunParamLimits) -> None:
    """校验五参均在配置范围内；越界则抛出 ``GunParamsOutOfRangeError``。"""
    checks: list[tuple[str, float, float, float, str]] = [
        ("阳极高压电压", params.anode_kv, limits.anode_kv_min, limits.anode_kv_max, "kV"),
        ("阳极高压电流", params.anode_ua, limits.anode_ua_min, limits.anode_ua_max, "uA"),
        ("阴极电压", params.cathode_v, limits.cathode_v_min, limits.cathode_v_max, "V"),
        ("栅偏电压", params.bias_v, limits.bias_v_min, limits.bias_v_max, "V"),
        ("灯丝电流", params.fil_a, limits.fil_a_min, limits.fil_a_max, "A"),
    ]
    lines: list[str] = []
    for label, value, lo, hi, unit in checks:
        if value < lo or value > hi:
            lines.append(
                f"警告: {label}={value}{unit} 超出允许范围 [{lo}, {hi}]{unit}"
            )
    if lines:
        raise GunParamsOutOfRangeError("\n".join(lines))
