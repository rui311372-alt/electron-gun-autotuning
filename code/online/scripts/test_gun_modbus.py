"""冒烟：连接、读反馈、可选写入（默认只读，加 --set 才写参）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gun_modbus import GunModbusClient, GunModbusError, GunParams, load_gun_modbus_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="电子枪 Modbus 冒烟测试")
    parser.add_argument("config", nargs="?", help="gun_modbus_config.json 路径")
    parser.add_argument(
        "--set",
        action="store_true",
        help="写入示例参数（危险：仅在你确认安全值时使用）",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser() if args.config else None
    try:
        cfg, used = load_gun_modbus_config(cfg_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"config: {used}")
    try:
        with GunModbusClient(cfg) as gun:
            if args.set:
                demo = GunParams(
                    anode_kv=90.0,
                    anode_ua=100.0,
                    cathode_v=660.0,
                    bias_v=160.0,
                    fil_a=0.46,
                )
                gun.set_params(demo)
                print("set_params OK, waiting bias stable…")
                if not gun.wait_bias_stable(target_v=demo.bias_v):
                    print("bias not stable within timeout", file=sys.stderr)
                    return 1
            fbk = gun.read_params()
            print(
                f"FBK  anode_kv={fbk.anode_kv}  anode_ua={fbk.anode_ua}  "
                f"cathode_v={fbk.cathode_v}  bias_v={fbk.bias_v}  fil_a={fbk.fil_a}"
            )
    except GunModbusError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())