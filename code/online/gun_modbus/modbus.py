"""电子枪 Modbus 串口基础接口（沿用原 ``project/modbus.py`` 逻辑，勿在 import 时连接）。"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pymodbus.client import ModbusSerialClient

if TYPE_CHECKING:
    pass

# 由 ``connect_serial`` 赋值；``set_value`` / ``read_value`` / ``set_state_on_or_off`` 直接使用
client: ModbusSerialClient | None = None

SLAVE_DEFAULT = 0x04

# 连接状态文件，用于跨进程判断连接状态
_CONNECTION_STATE_FILE = None

# pymodbus 3.x 用 device_id；2.x 用 slave（启动时探测一次）
_DEVICE_ID_KW: str | None = None


def _device_kw(slave: int) -> dict[str, int]:
    global _DEVICE_ID_KW
    if _DEVICE_ID_KW is None:
        import inspect

        from pymodbus.client.mixin import ModbusClientMixin

        params = inspect.signature(ModbusClientMixin.write_register).parameters
        _DEVICE_ID_KW = "device_id" if "device_id" in params else "slave"
    return {_DEVICE_ID_KW: slave}


def _response_error(response: Any) -> bool:
    if response is None:
        return True
    if hasattr(response, "isError"):
        return bool(response.isError())
    if hasattr(response, "is_error"):
        return bool(response.is_error())
    return False


class GunModbusError(RuntimeError):
    """未连接或读写失败。"""


def _require_client() -> ModbusSerialClient:
    if client is None:
        raise GunModbusError("Modbus 未连接，请先调用 connect_serial() 或 GunModbusClient.open()")
    return client


def _get_state_file() -> str:
    """获取连接状态文件路径。"""
    global _CONNECTION_STATE_FILE
    if _CONNECTION_STATE_FILE is None:
        import tempfile
        _CONNECTION_STATE_FILE = os.path.join(tempfile.gettempdir(), "gun_modbus_connection.state")
    return _CONNECTION_STATE_FILE


def connect_serial(
    port: str = "COM3",
    baudrate: int = 9600,
    bytesize: int = 8,
    parity: str = "N",
    stopbits: int = 1,
    timeout: float = 1.0,
) -> bool:
    """创建串口客户端并连接；成功返回 True。"""
    global client
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    client = ModbusSerialClient(
        port=port,
        baudrate=baudrate,
        bytesize=bytesize,
        parity=parity,
        stopbits=stopbits,
        timeout=timeout,
    )
    result = bool(client.connect())
    if result:
        # 记录连接状态
        with open(_get_state_file(), "w") as f:
            f.write(port)
    return result


def is_connected() -> bool:
    """检查是否已连接（通过状态文件判断）。"""
    state_file = _get_state_file()
    if not os.path.exists(state_file):
        return False
    # 检查客户端是否仍有效
    if client is None:
        return False
    return True


def get_connected_port() -> str | None:
    """获取当前连接的端口号。"""
    state_file = _get_state_file()
    if not os.path.exists(state_file):
        return None
    try:
        with open(state_file, "r") as f:
            return f.read().strip()
    except:
        return None


def disconnect_serial() -> None:
    global client
    if client is not None:
        client.close()
        client = None
    # 删除连接状态文件
    state_file = _get_state_file()
    if os.path.exists(state_file):
        os.remove(state_file)
    # 删除连接状态文件
    state_file = _get_state_file()
    if os.path.exists(state_file):
        os.remove(state_file)
    # 删除连接状态文件
    state_file = _get_state_file()
    if os.path.exists(state_file):
        os.remove(state_file)
    # 删除连接状态文件
    state_file = _get_state_file()
    if os.path.exists(state_file):
        os.remove(state_file)
    # 删除连接状态文件
    state_file = _get_state_file()
    if os.path.exists(state_file):
        os.remove(state_file)
    # 删除连接状态文件
    state_file = _get_state_file()
    if os.path.exists(state_file):
        os.remove(state_file)


def set_state_on_or_off(register_address: int, value: int = 1, *, slave: int = SLAVE_DEFAULT) -> None:
    """
    register_address: HV:1, BIAS:2, BEAM:3, FIL:4, Remote:10, N2K:11
    value: on=1, off=0
    """
    c = _require_client()
    write_hv_response = c.write_register(register_address, value, **_device_kw(slave))
    if _response_error(write_hv_response):
        raise GunModbusError(f"写入状态寄存器 {register_address} 失败: {write_hv_response}")


def set_value(value: float, mode: str, *, slave: int = SLAVE_DEFAULT) -> None:
    address = 0
    raw = value
    if mode == "FILAMENT":
        value = int(value * 52428 / 1.5)
        address = 5
    elif mode == "CURRENT":
        value = int(value * 52428 / 300)
        address = 6
    elif mode == "VOLTAGE":
        value = int(value * 52428 / 156)
        address = 7
    elif mode == "BIAS":
        value = int(value * 52428 / 200)
        address = 8
    elif mode == "N2K":
        value = int(value * 3276 / 2000)
        address = 9
    else:
        raise GunModbusError(f"未知 set mode: {mode!r}")
    result = _require_client().write_register(address, value, **_device_kw(slave))
    if _response_error(result):
        raise GunModbusError(f"写入 {mode}={raw} 失败: {result}")


def read_value(mode: str, *, slave: int = SLAVE_DEFAULT) -> float:
    address = 0
    w1 = 0
    w2 = 0.0
    if mode == "N2K IFBK":
        address = 2
        w1 = 3276
        w2 = 1
    elif mode == "DC5V FBK":
        address = 3
        w1 = 4095
        w2 = 5
    elif mode == "DC15V FBK":
        address = 4
        w1 = 2233
        w2 = 15
    elif mode == "DC24V FBK":
        address = 5
        w1 = 3573
        w2 = 24
    elif mode == "BIAS IFBK":
        address = 6
        w1 = 3276
        w2 = 1
    elif mode == "N2K VFBK":
        address = 7
        w1 = 3276
        w2 = 2000
    elif mode == "HV FBK":
        address = 8
        w1 = 52428
        w2 = 156
    elif mode == "mA FBK":
        address = 9
        w1 = 52428
        w2 = 300
    elif mode == "BIAS VFBK":
        address = 10
        w1 = 52428
        w2 = 200
    elif mode == "FIL IFBK":
        address = 11
        w1 = 52428
        w2 = 1.5
    elif mode == "FIL VFBK":
        address = 12
        w1 = 3276
        w2 = 8
    elif mode == "HV SPARK COUNT":
        address = 13
        w1 = 60000
        w2 = 60000
    elif mode == "N2K SPARK COUNT":
        address = 14
        w1 = 60000
        w2 = 60000
    else:
        raise GunModbusError(f"未知 read mode: {mode!r}")

    c = _require_client()
    rr = c.read_input_registers(address=address, count=1, **_device_kw(slave))
    if _response_error(rr):
        raise GunModbusError(f"读取 {mode} 失败: {rr}")
    return w2 * rr.registers[0] / w1
