"""``gun_modbus_config.json`` 加载。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .params import GunParamLimits


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    return project_root() / "gun_modbus_config.json"


def _strip_comments(obj: object) -> object:
    if isinstance(obj, dict):
        return {
            k: _strip_comments(v)
            for k, v in obj.items()
            if isinstance(k, str) and not (k == "_note" or k.endswith("_comment"))
        }
    if isinstance(obj, list):
        return [_strip_comments(x) for x in obj]
    return obj


@dataclass
class GunModbusConfig:
    port: str = "COM3"
    baudrate: int = 9600
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    timeout_s: float = 1.0
    slave: int = 0x04
    bias_stable_eps_v: float = 0.5
    bias_stable_samples: int = 3
    bias_stable_poll_s: float = 0.2
    bias_stable_timeout_s: float = 30.0
    param_limits: GunParamLimits = field(default_factory=GunParamLimits)


def _limits_from_dict(pl: dict) -> GunParamLimits:
    def _pair(key: str, lo_attr: str, hi_attr: str) -> tuple[float, float]:
        block = pl.get(key)
        defaults = GunParamLimits()
        if isinstance(block, (int, float)):
            v = float(block)
            return v, v
        if isinstance(block, dict):
            lo = float(block.get("min", getattr(defaults, lo_attr)))
            hi = float(block.get("max", getattr(defaults, hi_attr)))
        elif isinstance(block, list) and block:
            lo = float(min(block))
            hi = float(max(block))
        else:
            lo = float(getattr(defaults, lo_attr))
            hi = float(getattr(defaults, hi_attr))
        return lo, hi

    akv = _pair("anode_kv", "anode_kv_min", "anode_kv_max")
    aua = _pair("anode_ua", "anode_ua_min", "anode_ua_max")
    cv = _pair("cathode_v", "cathode_v_min", "cathode_v_max")
    bv = _pair("bias_v", "bias_v_min", "bias_v_max")
    fa = _pair("fil_a", "fil_a_min", "fil_a_max")
    return GunParamLimits(
        anode_kv_min=akv[0],
        anode_kv_max=akv[1],
        anode_ua_min=aua[0],
        anode_ua_max=aua[1],
        cathode_v_min=cv[0],
        cathode_v_max=cv[1],
        bias_v_min=bv[0],
        bias_v_max=bv[1],
        fil_a_min=fa[0],
        fil_a_max=fa[1],
    )


def load_gun_modbus_config(path: Path | None = None) -> tuple[GunModbusConfig, Path]:
    if path is not None:
        p = path.expanduser().resolve()
    else:
        env = os.environ.get("GUN_MODBUS_CONFIG", "").strip()
        p = Path(env).expanduser().resolve() if env else default_config_path()
    if not p.is_file():
        raise FileNotFoundError(f"gun modbus config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    d = _strip_comments(raw)
    if not isinstance(d, dict):
        d = {}
    serial = d.get("serial") if isinstance(d.get("serial"), dict) else {}
    bias = d.get("bias_stable") if isinstance(d.get("bias_stable"), dict) else {}
    pl = d.get("param_limits") if isinstance(d.get("param_limits"), dict) else {}
    cfg = GunModbusConfig(
        port=str(serial.get("port", GunModbusConfig.port)),
        baudrate=int(serial.get("baudrate", GunModbusConfig.baudrate)),
        bytesize=int(serial.get("bytesize", GunModbusConfig.bytesize)),
        parity=str(serial.get("parity", GunModbusConfig.parity)),
        stopbits=int(serial.get("stopbits", GunModbusConfig.stopbits)),
        timeout_s=float(serial.get("timeout_s", GunModbusConfig.timeout_s)),
        slave=int(d.get("slave", GunModbusConfig.slave)),
        bias_stable_eps_v=float(bias.get("eps_v", GunModbusConfig.bias_stable_eps_v)),
        bias_stable_samples=int(
            bias.get("samples", GunModbusConfig.bias_stable_samples)
        ),
        bias_stable_poll_s=float(
            bias.get("poll_s", GunModbusConfig.bias_stable_poll_s)
        ),
        bias_stable_timeout_s=float(
            bias.get("timeout_s", GunModbusConfig.bias_stable_timeout_s)
        ),
        param_limits=_limits_from_dict(pl),
    )
    return cfg, p