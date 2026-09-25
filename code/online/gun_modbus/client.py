"""在 ``modbus.py`` 基础接口之上的会话封装（PDF 五参读写、使能、FIL/BIAS 平稳）。"""

from __future__ import annotations

import importlib
import time

mb = importlib.import_module(".modbus", __package__)
from .config_io import GunModbusConfig, load_gun_modbus_config
from .params import (
    FBK_BIAS_V,
    FBK_FIL_I,
    FBK_HV,
    FBK_MA,
    FBK_N2K_V,
    GunParams,
    SET_BIAS,
    SET_CURRENT,
    SET_FILAMENT,
    SET_N2K,
    SET_VOLTAGE,
    STATE_BEAM,
    STATE_BIAS,
    STATE_FIL,
    STATE_HV,
    STATE_N2K,
    STATE_REMOTE,
)

# 上电顺序：先远控/低压路，最后 HV；下电相反
_STATE_ON_ORDER = (
    STATE_REMOTE,
    STATE_N2K,
    STATE_FIL,
    STATE_BIAS,
    STATE_BEAM,
    STATE_HV,
)
_STATE_OFF_ORDER = (
    STATE_HV,
    STATE_BEAM,
    STATE_BIAS,
    STATE_N2K,
    STATE_REMOTE,
)


class GunModbusClient:
    """打开时 ``connect_serial``，关闭时 ``disconnect_serial``；读写直接调用 ``gun_modbus.modbus``。"""

    def __init__(self, cfg: GunModbusConfig | None = None) -> None:
        if cfg is None:
            cfg, _ = load_gun_modbus_config()
        self._cfg = cfg
        self._slave = int(cfg.slave)

    def __enter__(self) -> GunModbusClient:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def open(self) -> None:
        ok = mb.connect_serial(
            port=self._cfg.port,
            baudrate=self._cfg.baudrate,
            bytesize=self._cfg.bytesize,
            parity=self._cfg.parity,
            stopbits=self._cfg.stopbits,
            timeout=self._cfg.timeout_s,
        )
        if not ok:
            raise mb.GunModbusError(f"无法连接串口 {self._cfg.port}")

    def close(self) -> None:
        mb.disconnect_serial()

    def set_state_on_or_off(self, register_address: int, value: int = 1) -> None:
        mb.set_state_on_or_off(register_address, value, slave=self._slave)

    def set_all_on(self) -> None:
        """远控 + N2K/FIL/BIAS/BEAM/HV 依次置 on（与旧 modbus 寄存器表一致）。"""
        for addr in _STATE_ON_ORDER:
            self.set_state_on_or_off(addr, 1)

    def set_all_off(self) -> None:
        """HV 先关，其余依次 off。"""
        for addr in _STATE_OFF_ORDER:
            self.set_state_on_or_off(addr, 0)

    def set_value(self, value: float, mode: str) -> None:
        mb.set_value(value, mode, slave=self._slave)

    def read_value(self, mode: str) -> float:
        return mb.read_value(mode, slave=self._slave)

    def set_params(self, params: GunParams) -> None:
        """按 PDF 顺序写入五路设定值。"""
        mb.set_value(params.anode_kv, SET_VOLTAGE, slave=self._slave)
        mb.set_value(params.bias_v, SET_BIAS, slave=self._slave)
        mb.set_value(params.cathode_v, SET_N2K, slave=self._slave)
        mb.set_value(params.fil_a, SET_FILAMENT, slave=self._slave)
        mb.set_value(params.anode_ua, SET_CURRENT, slave=self._slave)

    def read_params(self) -> GunParams:
        """从各 FBK 回读当前反馈（整型换算，与旧脚本一致）。"""
        return GunParams(
            anode_kv=float(mb.read_value(FBK_HV, slave=self._slave)),
            anode_ua=float(mb.read_value(FBK_MA, slave=self._slave)),
            cathode_v=float(mb.read_value(FBK_N2K_V, slave=self._slave)),
            bias_v=float(mb.read_value(FBK_BIAS_V, slave=self._slave)),
            fil_a=float(mb.read_value(FBK_FIL_I, slave=self._slave)),
        )

    def _wait_feedback_stable(
        self,
        fbk_mode: str,
        target: float | None,
        *,
        eps: float,
        samples: int,
        poll_s: float,
        timeout_s: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        stable_count = 0
        last: float | None = None
        while time.monotonic() < deadline:
            v = float(mb.read_value(fbk_mode, slave=self._slave))
            if target is not None:
                ok = abs(v - target) <= eps
            elif last is not None:
                ok = abs(v - last) <= eps
            else:
                ok = False
            last = v
            if ok:
                stable_count += 1
                if stable_count >= samples:
                    return True
            else:
                stable_count = 0
            time.sleep(poll_s)
        return False

    def wait_bias_stable(
        self,
        target_v: float | None = None,
        *,
        eps_v: float | None = None,
        samples: int | None = None,
        poll_s: float | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """轮询 ``BIAS VFBK``，连续 ``samples`` 次 |读数-目标|<=``eps_v`` 则视为平稳。"""
        return self._wait_feedback_stable(
            FBK_BIAS_V,
            target_v,
            eps=eps_v if eps_v is not None else self._cfg.bias_stable_eps_v,
            samples=samples if samples is not None else self._cfg.bias_stable_samples,
            poll_s=poll_s if poll_s is not None else self._cfg.bias_stable_poll_s,
            timeout_s=(
                timeout_s if timeout_s is not None else self._cfg.bias_stable_timeout_s
            ),
        )