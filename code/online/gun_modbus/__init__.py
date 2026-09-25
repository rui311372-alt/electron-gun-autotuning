"""电子枪 Modbus 控制：基础接口在 ``modbus.py``（原 project/modbus.py），上层用 ``GunModbusClient``。"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .config_io import GunModbusConfig, default_config_path, load_gun_modbus_config
from .params import (
    DEFAULT_BIAS_V,
    GunParamLimits,
    GunParams,
    GunParamsOutOfRangeError,
    validate_gun_params,
)

if TYPE_CHECKING:
    from .client import GunModbusClient
    from .modbus import GunModbusError

__all__ = [
    "GunModbusClient",
    "GunModbusConfig",
    "GunModbusError",
    "DEFAULT_BIAS_V",
    "GunParamLimits",
    "GunParams",
    "GunParamsOutOfRangeError",
    "validate_gun_params",
    "connect_serial",
    "default_config_path",
    "disconnect_serial",
    "load_gun_modbus_config",
    "modbus",
    "read_value",
    "set_state_on_or_off",
    "set_value",
]


def _modbus_module():
    """加载子模块 ``modbus``（勿在 ``__getattr__`` 里 ``from . import modbus``，会递归）。"""
    return importlib.import_module(".modbus", __name__)


def __getattr__(name: str):
    if name == "GunModbusClient":
        from .client import GunModbusClient

        return GunModbusClient
    if name == "GunModbusError":
        return _modbus_module().GunModbusError
    if name in ("connect_serial", "disconnect_serial", "read_value", "set_state_on_or_off", "set_value"):
        return getattr(_modbus_module(), name)
    if name == "modbus":
        return _modbus_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
