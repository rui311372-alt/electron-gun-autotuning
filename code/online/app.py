"""电子枪闭环：采图 → 算分(目标函数) → 设参 → FIL/BIAS 平稳 → 采图，循环直至优化结束。

单次迭代（供贝叶斯等优化器反复调用）::

    from app import ClosedLoopSession, evaluate

    with ClosedLoopSession() as loop:
        r = evaluate(loop, params_next)   # 返回本轮 score（对应当前硬件状态）
        # 优化器根据 r.score 提议下一组 params_next

命令行::

    python app.py run --kv 90 --ua 100 --cv 660 --fa 0.46   # 单步（--bv 默认 200）
    python app.py optimize --params-sequence steps.json --max-iters 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.config_schema import (  # noqa: E402
    AnalysisConfig,
    load_analysis_config,
    resolve_single_save_debug_path,
    resolve_under_config_dir,
)
from analysis.io_tif import load_tif14  # noqa: E402
from analysis.score import ScoreResult, compute_score  # noqa: E402
from analysis.visualize import save_debug_figure  # noqa: E402
from camera_cap.config_io import CameraCaptureConfig, load_camera_config  # noqa: E402
from camera_cap.frame_names import single_tif_path  # noqa: E402
from camera_cap.soft_trigger import CameraRuntimeError, SoftTriggerSession  # noqa: E402
from gun_modbus.config_io import GunModbusConfig, load_gun_modbus_config  # noqa: E402
from gun_modbus.params import (  # noqa: E402
    DEFAULT_BIAS_V,
    GunParams,
    GunParamsOutOfRangeError,
    SET_BIAS,
    SET_CURRENT,
    SET_FILAMENT,
    SET_N2K,
    SET_VOLTAGE,
    validate_gun_params,
)


@dataclass
class AppRunResult:
    """一次闭环迭代的结果。score 来自本轮开头的采图（上一迭代设参后的状态）。"""

    ok: bool
    score: float
    x1: float
    x2: float
    reason: str
    tif_scored: Path | None = None
    tif_prepared: Path | None = None
    debug_path: Path | None = None
    params_set: GunParams | None = None
    params_read: GunParams | None = None

    @property
    def tif_path(self) -> Path | None:
        return self.tif_scored


def _capture(cam_cfg: CameraCaptureConfig, tif_out: Path) -> Path:
    with SoftTriggerSession(cam_cfg) as sess:
        return sess.grab_to_tif(tif_out)


def analyze_tif(
    tif_path: Path,
    analysis_cfg: AnalysisConfig,
    analysis_cfg_path: Path,
    *,
    save_debug: bool = True,
) -> tuple[ScoreResult, Path | None]:
    img = load_tif14(tif_path)
    result = compute_score(img, config=analysis_cfg)
    debug_path: Path | None = None
    if save_debug:
        save_raw = (analysis_cfg.io.save_debug_path or "").strip()
        if not save_raw:
            raise ValueError("analysis_config.io.save_debug_path 未配置")
        debug_path = resolve_single_save_debug_path(
            analysis_cfg_path, save_raw, tif_path
        )
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        save_debug_figure(img, result, debug_path, title=tif_path.name)
    return result, debug_path


def optimization_step(
    params_next: GunParams | None,
    *,
    cam_cfg: CameraCaptureConfig,
    ana_cfg: AnalysisConfig,
    ana_cfg_path: Path,
    gun: Any = None,
    skip_modbus: bool = False,
    skip_capture: bool = False,
    wait_bias: bool = True,
    save_debug: bool = True,
) -> AppRunResult:
    """一次闭环：采图 → compute_score → set_params → wait_bias_stable → 采图。

    Modbus 须由 ``ClosedLoopSession.open_modbus()`` 在循环外打开并使能；本函数只做写参/等稳/读反馈。

    返回的 ``score`` 对应当前硬件上的光斑（上一迭代末尾 ``set_params`` 之后的状态；
    首次调用则为启动时的状态）。本轮末尾写入 ``params_next`` 并再采一张图，供下一轮算分。
    """
    tif_out = single_tif_path((ROOT / cam_cfg.output_dir.strip()).resolve())
    params_read: GunParams | None = None
    tif_scored: Path | None = None
    tif_prepared: Path | None = None

    if not skip_capture:
        try:
            tif_scored = _capture(cam_cfg, tif_out)
        except CameraRuntimeError as exc:
            return AppRunResult(
                ok=False,
                score=float("nan"),
                x1=float("nan"),
                x2=float("nan"),
                reason=f"capture_error:{exc}",
                params_set=params_next,
            )
    else:
        rel = (ana_cfg.io.single_image or "").strip()
        if rel:
            tif_scored = resolve_under_config_dir(ana_cfg_path, rel)
        if tif_scored is None or not tif_scored.is_file():
            return AppRunResult(
                ok=False,
                score=float("nan"),
                x1=float("nan"),
                x2=float("nan"),
                reason=f"tif_not_found:{tif_scored}",
                params_set=params_next,
            )

    try:
        score_result, debug_path = analyze_tif(
            tif_scored, ana_cfg, ana_cfg_path, save_debug=save_debug
        )
    except Exception as exc:
        return AppRunResult(
            ok=False,
            score=float("nan"),
            x1=float("nan"),
            x2=float("nan"),
            reason=f"analyze_error:{exc}",
            tif_scored=tif_scored,
            params_set=params_next,
        )

    if not skip_modbus:
        if params_next is None:
            raise ValueError("未提供 params_next")
        if gun is None:
            raise ValueError(
                "skip_modbus=False 时需要已打开的 GunModbusClient；请用 ClosedLoopSession"
            )
        gun.set_params(params_next)
        if wait_bias:
            if not gun.wait_bias_stable(target_v=params_next.bias_v):
                return AppRunResult(
                    ok=False,
                    score=float(score_result.score),
                    x1=float(score_result.x1),
                    x2=float(score_result.x2),
                    reason="bias_not_stable",
                    tif_scored=tif_scored,
                    debug_path=debug_path,
                    params_set=params_next,
                )
        params_read = gun.read_params()

    if not skip_capture and not skip_modbus:
        try:
            tif_prepared = _capture(cam_cfg, tif_out)
        except CameraRuntimeError as exc:
            return AppRunResult(
                ok=False,
                score=float(score_result.score),
                x1=float(score_result.x1),
                x2=float(score_result.x2),
                reason=f"capture_after_set_error:{exc}",
                tif_scored=tif_scored,
                debug_path=debug_path,
                params_set=params_next,
                params_read=params_read,
            )

    return AppRunResult(
        ok=score_result.ok,
        score=float(score_result.score),
        x1=float(score_result.x1),
        x2=float(score_result.x2),
        reason=score_result.reason,
        tif_scored=tif_scored,
        tif_prepared=tif_prepared,
        debug_path=debug_path,
        params_set=params_next,
        params_read=params_read,
    )


@dataclass
class ClosedLoopSession:
    """加载配置后可反复调用 ``step`` / ``run_until``。

    Modbus：``open_modbus`` 在循环开始前连接并使能一次，``close_modbus`` / 上下文退出时 OFF 并断开；
    各 ``step`` 仅复用同一 ``GunModbusClient`` 写参/等稳/读反馈。
    """

    gun_cfg: GunModbusConfig | None = None
    gun_cfg_path: Path | None = None
    cam_cfg: CameraCaptureConfig = field(default_factory=CameraCaptureConfig)
    ana_cfg: AnalysisConfig = field(default_factory=AnalysisConfig)
    ana_cfg_path: Path = field(default_factory=lambda: ROOT / "analysis_config.json")
    cam_cfg_path: Path = field(default_factory=lambda: ROOT / "camera_capture_config.json")
    history: list[AppRunResult] = field(default_factory=list)
    _gun: Any = field(default=None, repr=False, compare=False)

    def __enter__(self) -> ClosedLoopSession:
        self.load()
        return self

    def __exit__(self, *args: object) -> None:
        self.close_modbus()

    def open_modbus(self, *, skip_modbus: bool = False) -> None:
        """连接串口并使能 ON（已打开则跳过）。"""
        if skip_modbus or self._gun is not None:
            return
        if self.gun_cfg is None:
            raise ValueError("未找到 gun_modbus 配置，无法打开 Modbus")
        from gun_modbus.client import GunModbusClient

        self._gun = GunModbusClient(self.gun_cfg)
        self._gun.open()
        self._gun.set_all_on()

    def close_modbus(self) -> None:
        """使能 OFF 并断开串口（未打开则跳过）。"""
        gun = self._gun
        if gun is None:
            return
        self._gun = None
        from gun_modbus.modbus import GunModbusError

        try:
            gun.set_all_off()
        except GunModbusError:
            pass
        finally:
            gun.close()

    def load(
        self,
        gun_cfg_path: Path | None = None,
        camera_cfg_path: Path | None = None,
        analysis_cfg_path: Path | None = None,
    ) -> None:
        self.cam_cfg, self.cam_cfg_path = load_camera_config(camera_cfg_path)
        self.ana_cfg, self.ana_cfg_path = load_analysis_config(analysis_cfg_path)
        try:
            self.gun_cfg, self.gun_cfg_path = load_gun_modbus_config(gun_cfg_path)
        except FileNotFoundError:
            self.gun_cfg = None
            self.gun_cfg_path = None

    def step(
        self,
        params_next: GunParams,
        *,
        skip_modbus: bool = False,
        skip_capture: bool = False,
        wait_bias: bool = True,
        save_debug: bool = True,
    ) -> AppRunResult:
        if not skip_modbus:
            try:
                _check_params_range(params_next, self.gun_cfg, context="五参越界")
            except GunParamsOutOfRangeError as exc:
                r = _params_out_of_range_result(params_next, reason_detail=str(exc))
                self.history.append(r)
                return r
        self.open_modbus(skip_modbus=skip_modbus)
        r = optimization_step(
            params_next,
            cam_cfg=self.cam_cfg,
            ana_cfg=self.ana_cfg,
            ana_cfg_path=self.ana_cfg_path,
            gun=self._gun,
            skip_modbus=skip_modbus,
            skip_capture=skip_capture,
            wait_bias=wait_bias,
            save_debug=save_debug,
        )
        self.history.append(r)
        return r

    def run_until(
        self,
        suggest: Callable[[list[AppRunResult]], GunParams | None],
        *,
        max_iters: int = 100,
        target_score: float | None = None,
        require_ok: bool = True,
        skip_modbus: bool = False,
        skip_capture: bool = False,
        wait_bias: bool = True,
        save_debug: bool = True,
        final_score: bool = True,
    ) -> list[AppRunResult]:
        """循环：采图→算分→设参→等 BIAS 稳→采图，直到 ``suggest`` 返回 None 或达到停止条件。"""
        self.open_modbus(skip_modbus=skip_modbus)
        out: list[AppRunResult] = []
        try:
            for _ in range(max_iters):
                params = suggest(self.history)
                if params is None:
                    break
                r = self.step(
                    params,
                    skip_modbus=skip_modbus,
                    skip_capture=skip_capture,
                    wait_bias=wait_bias,
                    save_debug=save_debug,
                )
                out.append(r)
                if require_ok and not r.ok:
                    break
                if target_score is not None and r.ok and r.score <= target_score:
                    break

            if final_score and out and not skip_capture:
                last = self._final_score_only(save_debug=save_debug)
                if last is not None:
                    out.append(last)
            return out
        finally:
            self.close_modbus()

    def _final_score_only(self, *, save_debug: bool) -> AppRunResult | None:
        """最后一轮设参后的图在本轮未算分；补一次仅采图+算分。"""
        tif_out = single_tif_path((ROOT / self.cam_cfg.output_dir.strip()).resolve())
        try:
            tif_scored = _capture(self.cam_cfg, tif_out)
        except CameraRuntimeError:
            return None
        try:
            score_result, debug_path = analyze_tif(
                tif_scored, self.ana_cfg, self.ana_cfg_path, save_debug=save_debug
            )
        except Exception:
            return None
        r = AppRunResult(
            ok=score_result.ok,
            score=float(score_result.score),
            x1=float(score_result.x1),
            x2=float(score_result.x2),
            reason=score_result.reason or "final_score",
            tif_scored=tif_scored,
            debug_path=debug_path,
        )
        self.history.append(r)
        return r


def evaluate(
    session: ClosedLoopSession,
    params_next: GunParams,
    **kwargs: Any,
) -> AppRunResult:
    """优化器目标函数一步：等价于 ``session.step(params_next)``。"""
    return session.step(params_next, **kwargs)


def run_once(
    params: GunParams | None = None,
    *,
    skip_modbus: bool = False,
    skip_capture: bool = False,
    wait_bias: bool = True,
    save_debug: bool = True,
    gun_cfg_path: Path | None = None,
    camera_cfg_path: Path | None = None,
    analysis_cfg_path: Path | None = None,
) -> AppRunResult:
    """单次闭环迭代（兼容旧名）。"""
    if params is None and not skip_modbus:
        raise ValueError("未提供 GunParams；设参闭环须给出下一组五参")
    with ClosedLoopSession() as sess:
        sess.load(gun_cfg_path, camera_cfg_path, analysis_cfg_path)
        return sess.step(
            params,
            skip_modbus=skip_modbus,
            skip_capture=skip_capture,
            wait_bias=wait_bias,
            save_debug=save_debug,
        )


def _params_from_dict(
    d: dict[str, Any],
    *,
    default_bias_v: float | None = None,
) -> GunParams:
    if "bias_v" in d:
        bias_v = float(d["bias_v"])
    elif default_bias_v is not None:
        bias_v = default_bias_v
    else:
        raise KeyError("bias_v")
    return GunParams(
        anode_kv=float(d["anode_kv"]),
        anode_ua=float(d["anode_ua"]),
        cathode_v=float(d["cathode_v"]),
        bias_v=bias_v,
        fil_a=float(d["fil_a"]),
    )


def _params_from_args(args: argparse.Namespace) -> GunParams | None:
    if getattr(args, "params_json", None):
        p = Path(args.params_json).expanduser()
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            raise ValueError("单步 run 请使用单个五参对象；多步请用 optimize --params-sequence")
        return _params_from_dict(data, default_bias_v=DEFAULT_BIAS_V)
    # 若提供了除 Ug 外的四参，Ug 默认用 200
    ug = getattr(args, "Ug", None)
    if ug is None:
        other_keys = ("Ua", "Ia", "Uc", "If")
        if all(getattr(args, k, None) is not None for k in other_keys):
            ug = DEFAULT_BIAS_V
    if all(getattr(args, k, None) is not None for k in ("Ua", "Ia", "Uc", "If")) and ug is not None:
        return GunParams(
            anode_kv=float(args.Ua),
            anode_ua=float(args.Ia),
            cathode_v=float(args.Uc),
            bias_v=float(ug),
            fil_a=float(args.If),
        )
    return None


def _load_params_sequence(path: Path) -> list[GunParams]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("params-sequence 须为 JSON 数组，每项含五参字段")
    return [_params_from_dict(item, default_bias_v=DEFAULT_BIAS_V) for item in data]


def _check_params_range(
    params: GunParams,
    gun_cfg: GunModbusConfig | None,
    *,
    gun_cfg_path: Path | None = None,
    context: str = "",
) -> None:
    """越界则打印警告到 stderr 并抛出 GunParamsOutOfRangeError。"""
    if gun_cfg is None:
        gun_cfg, _ = load_gun_modbus_config(gun_cfg_path)
    try:
        validate_gun_params(params, gun_cfg.param_limits)
    except GunParamsOutOfRangeError as exc:
        prefix = f"{context}: " if context else ""
        print(f"{prefix}{exc}", file=sys.stderr)
        raise


def _params_out_of_range_result(
    params: GunParams,
    *,
    reason_detail: str,
) -> AppRunResult:
    return AppRunResult(
        ok=False,
        score=float("nan"),
        x1=float("nan"),
        x2=float("nan"),
        reason=f"params_out_of_range:{reason_detail}",
        params_set=params,
    )


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gun-config", type=Path, default=None, help="gun_modbus_config.json")
    p.add_argument("--camera-config", type=Path, default=None, help="camera_capture_config.json")
    p.add_argument(
        "--analysis-config", type=Path, default=None, help="analysis_config.json"
    )


def _add_param_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--params-json", type=Path, default=None, help="五参 JSON 文件")
    p.add_argument("--Ua", type=float, default=None, help="ANODE V (kV)")
    p.add_argument("--Ia", type=float, default=None, help="ANODE I (uA)")
    p.add_argument("--Uc", type=float, default=None, help="CATHODE V")
    p.add_argument(
        "--Ug",
        type=float,
        default=None,
        help=f"BIAS V（默认 {DEFAULT_BIAS_V:g}）",
    )
    p.add_argument("--If", type=float, default=None, help="FIL I (A)")


def _print_run_result(r: AppRunResult, *, iteration: int | None = None) -> None:
    if iteration is not None:
        print(f"--- iter {iteration} ---")
    if r.params_set is not None:
        p = r.params_set
        print(
            f"set_params    : Ua={p.anode_kv}kV  Ia={p.anode_ua}uA  "
            f"Uc={p.cathode_v}V  Ug={p.bias_v}V  If={p.fil_a}A"
        )
    if r.params_read is not None:
        p = r.params_read
        print(
            f"read_params   : Ua={p.anode_kv}  Ia={p.anode_ua}  "
            f"Uc={p.cathode_v}  Ug={p.bias_v}  If={p.fil_a}"
        )
    if r.tif_scored is not None:
        print(f"tif_scored    : {r.tif_scored}")
    if r.tif_prepared is not None:
        print(f"tif_prepared  : {r.tif_prepared}")
    print(f"ok            : {r.ok}")
    print(f"x1 / x2       : {r.x1:.4f}  /  {r.x2:.4f}")
    print(f"score         : {r.score:.4f}")
    if r.reason:
        print(f"reason        : {r.reason}")
    if r.debug_path is not None:
        print(f"debug         : {r.debug_path}")


def cmd_run(args: argparse.Namespace) -> int:
    params = _params_from_args(args)
    if not args.skip_modbus and params is None:
        print(
            "请通过 --kv --ua --cv --fa（--bv 可选，默认 200）或 --params-json 提供下一组五参，或使用 --skip-modbus",
            file=sys.stderr,
        )
        return 2
    try:
        if not args.skip_modbus and params is not None:
            gun_cfg, _ = load_gun_modbus_config(args.gun_config)
            _check_params_range(params, gun_cfg, context="五参越界")
        r = run_once(
            params,
            skip_modbus=args.skip_modbus,
            skip_capture=args.skip_capture,
            wait_bias=not args.no_wait_bias,
            save_debug=not args.no_debug,
            gun_cfg_path=args.gun_config,
            camera_cfg_path=args.camera_config,
            analysis_cfg_path=args.analysis_config,
        )
    except GunParamsOutOfRangeError:
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        from gun_modbus.modbus import GunModbusError

        if isinstance(exc, GunModbusError):
            print(str(exc), file=sys.stderr)
            return 2
        raise
    _print_run_result(r)
    return 0 if r.ok else 1


def cmd_optimize(args: argparse.Namespace) -> int:
    if not args.params_sequence:
        print("请指定 --params-sequence（JSON 数组，每元素为一组五参）", file=sys.stderr)
        return 2
    try:
        sequence = _load_params_sequence(Path(args.params_sequence))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not sequence:
        print("params-sequence 为空", file=sys.stderr)
        return 2

    if not args.skip_modbus:
        try:
            gun_cfg, _ = load_gun_modbus_config(args.gun_config)
            for i, p in enumerate(sequence, start=1):
                _check_params_range(
                    p, gun_cfg, context=f"params-sequence 第 {i} 组越界"
                )
        except GunParamsOutOfRangeError:
            return 2

    idx = 0

    def suggest(history: list[AppRunResult]) -> GunParams | None:
        nonlocal idx
        if idx >= len(sequence) or idx >= args.max_iters:
            return None
        p = sequence[idx]
        idx += 1
        return p

    try:
        with ClosedLoopSession() as sess:
            sess.load(args.gun_config, args.camera_config, args.analysis_config)
            results = sess.run_until(
                suggest,
                max_iters=args.max_iters,
                target_score=args.target_score,
                require_ok=not args.continue_on_fail,
                skip_modbus=args.skip_modbus,
                skip_capture=args.skip_capture,
                wait_bias=not args.no_wait_bias,
                save_debug=not args.no_debug,
                final_score=not args.no_final_score,
            )
    except GunParamsOutOfRangeError:
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        from gun_modbus.modbus import GunModbusError

        if isinstance(exc, GunModbusError):
            print(str(exc), file=sys.stderr)
            return 2
        raise

    for i, r in enumerate(results, start=1):
        _print_run_result(r, iteration=i)
        print()

    if not results:
        return 1
    last = results[-1]
    print(f"stopped        : {len(results)} record(s), last score={last.score:.4f}")
    return 0 if last.ok else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        ana_cfg, ana_path = load_analysis_config(args.analysis_config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rel = (ana_cfg.io.single_image or "").strip()
    if not rel:
        print("analysis_config.io.single_image 未配置", file=sys.stderr)
        return 2
    tif_path = resolve_under_config_dir(ana_path, rel)
    if not tif_path.is_file():
        print(f"图像不存在: {tif_path}", file=sys.stderr)
        return 2
    try:
        result, debug_path = analyze_tif(
            tif_path, ana_cfg, ana_path, save_debug=not args.no_debug
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"config        : {ana_path}")
    print(f"image         : {tif_path.name}")
    print(f"ok            : {result.ok}")
    print(f"x1 / x2       : {result.x1:.4f}  /  {result.x2:.4f}")
    print(f"score         : {result.score:.4f}")
    if result.reason:
        print(f"reason        : {result.reason}")
    if debug_path:
        print(f"debug         : {debug_path}")
    return 0 if result.ok else 1


def cmd_capture(args: argparse.Namespace) -> int:
    try:
        cam_cfg, _ = load_camera_config(args.camera_config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        with SoftTriggerSession(cam_cfg) as sess:
            path = sess.grab_to_tif()
    except CameraRuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"tif           : {path}")
    return 0


MODBUS_SESSION_MARK = ROOT / ".gun_modbus_session"


def _modbus_session_mark(config_path: Path, port: str) -> None:
    MODBUS_SESSION_MARK.write_text(
        json.dumps({"config": str(config_path), "port": port}, ensure_ascii=False),
        encoding="utf-8",
    )


def _modbus_session_clear() -> None:
    MODBUS_SESSION_MARK.unlink(missing_ok=True)


def _modbus_session_require() -> None:
    from gun_modbus.modbus import GunModbusError

    if not MODBUS_SESSION_MARK.is_file():
        raise GunModbusError("请先运行: python app.py modbus-on")


def cmd_modbus_on(args: argparse.Namespace) -> int:
    from gun_modbus.client import GunModbusClient
    from gun_modbus.modbus import GunModbusError

    try:
        gun_cfg, used = load_gun_modbus_config(args.gun_config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    gun = GunModbusClient(gun_cfg)
    try:
        gun.open()
        print(f"config        : {used}")
        print(f"port          : {gun_cfg.port}  (connected)")
        gun.set_all_on()
        print("states        : Remote/N2K/FIL/BIAS/BEAM/HV ON")
        _modbus_session_mark(used, gun_cfg.port)
        print("note          : 连接已保持，可使用 modbus-set/modbus-read")
    except GunModbusError as exc:
        print(str(exc), file=sys.stderr)
        gun.close()
        return 1
    # 注意：这里不再主动 close，保持连接
    return 0


def cmd_modbus_off(args: argparse.Namespace) -> int:
    from gun_modbus.client import GunModbusClient
    from gun_modbus.modbus import GunModbusError, is_connected

    try:
        gun_cfg, used = load_gun_modbus_config(args.gun_config)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    gun = GunModbusClient(gun_cfg)
    code = 0
    try:
        # 如果已有连接，直接操作；否则重新连接
        if not is_connected():
            gun.open()
            print(f"port          : {gun_cfg.port}  (reconnected)")
        print(f"config        : {used}")
        gun.set_all_off()
        print("states        : HV/BEAM/BIAS/FIL/N2K/Remote OFF")
    except GunModbusError as exc:
        print(str(exc), file=sys.stderr)
        code = 1
    finally:
        gun.close()
        _modbus_session_clear()
        print("port          : disconnected")
    return code


def cmd_modbus_read(args: argparse.Namespace) -> int:
    from gun_modbus.client import GunModbusClient
    from gun_modbus.modbus import GunModbusError, is_connected

    try:
        gun_cfg, used = load_gun_modbus_config(args.gun_config)
        _modbus_session_require()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except GunModbusError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    gun = GunModbusClient(gun_cfg)
    try:
        # 复用已有连接，避免重复打开
        if not is_connected():
            gun.open()
            print(f"port          : {gun_cfg.port}  (connected)")
        print(f"config        : {used}")
        p = gun.read_params()
        print(
            f"FBK           : Ua={p.anode_kv}  Ia={p.anode_ua}  "
            f"Uc={p.cathode_v}  Ug={p.bias_v}  If={p.fil_a}"
        )
    except GunModbusError as exc:
        print(str(exc), file=sys.stderr)
        gun.close()
        return 1
    # 注意：这里不再主动 close，保持连接供后续使用
    return 0


def cmd_modbus_set(args: argparse.Namespace) -> int:
    from gun_modbus.client import GunModbusClient
    from gun_modbus.modbus import GunModbusError, is_connected

    try:
        params = _params_from_args(args)
    except (ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # 用户显式提供的参数
    provided = {
        k: float(getattr(args, k))
        for k in ("Ua", "Ia", "Uc", "Ug", "If")
        if getattr(args, k, None) is not None
    }

    if params is None and not provided:
        print(
            "请提供至少一个参数（--Ua --Ia --Uc --Ug --If），或完整五参，或 --params-json",
            file=sys.stderr,
        )
        return 2
    try:
        gun_cfg, used = load_gun_modbus_config(args.gun_config)
        _modbus_session_require()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except GunModbusError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    gun = GunModbusClient(gun_cfg)
    code = 0
    try:
        # 复用已有连接，避免重复打开
        if not is_connected():
            gun.open()
            print(f"port          : {gun_cfg.port}  (connected)")
        print(f"config        : {used}")

        if params is not None:
            # 全参设参
            _check_params_range(params, gun_cfg, context="五参越界")
            gun.set_params(params)
            print(
                f"set_params    : Ua={params.anode_kv}kV  Ia={params.anode_ua}uA  "
                f"Uc={params.cathode_v}V  Ug={params.bias_v}V  If={params.fil_a}A"
            )
            if args.wait_bias:
                if not gun.wait_bias_stable(target_v=params.bias_v):
                    print("bias not stable within timeout", file=sys.stderr)
                    code = 1
                else:
                    print("bias          : stable")
        else:
            # 部分设参：只写入用户提供的参数，不读取反馈填充
            param_info = {
                "Ua": (SET_VOLTAGE, "阳极高压电压", "kV", "anode_kv"),
                "Ia": (SET_CURRENT, "阳极高压电流", "uA", "anode_ua"),
                "Uc": (SET_N2K, "阴极电压", "V", "cathode_v"),
                "Ug": (SET_BIAS, "栅偏电压", "V", "bias_v"),
                "If": (SET_FILAMENT, "灯丝电流", "A", "fil_a"),
            }
            limit_map = {
                "Ua": ("anode_kv_min", "anode_kv_max"),
                "Ia": ("anode_ua_min", "anode_ua_max"),
                "Uc": ("cathode_v_min", "cathode_v_max"),
                "Ug": ("bias_v_min", "bias_v_max"),
                "If": ("fil_a_min", "fil_a_max"),
            }
            limits = gun_cfg.param_limits
            errors: list[str] = []
            for key, value in provided.items():
                lo = getattr(limits, limit_map[key][0])
                hi = getattr(limits, limit_map[key][1])
                if value < lo or value > hi:
                    label = param_info[key][1]
                    unit = param_info[key][2]
                    errors.append(
                        f"警告: {label}={value}{unit} 超出允许范围 [{lo}, {hi}]{unit}"
                    )
            if errors:
                raise GunParamsOutOfRangeError("\n".join(errors))

            parts: list[str] = []
            for key, value in provided.items():
                mode = param_info[key][0]
                gun.set_value(value, mode)
                label = param_info[key][1]
                unit = param_info[key][2]
                parts.append(f"{label}={value}{unit}")
            print(f"partial set   : {'  '.join(parts)}")

            if args.wait_bias and "Ug" in provided:
                if not gun.wait_bias_stable(target_v=provided["Ug"]):
                    print("bias not stable within timeout", file=sys.stderr)
                    code = 1
                else:
                    print("bias          : stable")

        if code == 0:
            fbk = gun.read_params()
            print(
                f"FBK           : Ua={fbk.anode_kv}  Ia={fbk.anode_ua}  "
                f"Uc={fbk.cathode_v}  Ug={fbk.bias_v}  If={fbk.fil_a}"
            )
    except GunParamsOutOfRangeError as exc:
        print(str(exc), file=sys.stderr)
        gun.close()
        return 2
    except GunModbusError as exc:
        print(str(exc), file=sys.stderr)
        gun.close()
        code = 1
    # 注意：这里不再主动 close，保持连接供后续使用
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="电子枪闭环：采图→算分→设参→FIL/BIAS 平稳→采图",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run",
        help="单步闭环：采图→算分→写五参→等 FIL/BIAS 稳→采图",
    )
    _add_config_args(p_run)
    _add_param_args(p_run)
    p_run.add_argument("--skip-modbus", action="store_true", help="跳过写电参（仅采图+算分）")
    p_run.add_argument("--skip-capture", action="store_true", help="不采图，分析已有 TIF")
    p_run.add_argument("--no-wait-bias", action="store_true", help="设 BIAS 后不等待平稳")
    p_run.add_argument("--no-debug", action="store_true", help="不保存调试图 PNG")
    p_run.set_defaults(func=cmd_run)

    p_opt = sub.add_parser(
        "optimize",
        help="按 params-sequence 循环直至步数/目标 score 用尽",
    )
    _add_config_args(p_opt)
    p_opt.add_argument(
        "--params-sequence",
        type=Path,
        required=True,
        help="JSON 数组，每元素一组五参（每轮 set_params 用）",
    )
    p_opt.add_argument("--max-iters", type=int, default=100, help="最大迭代次数")
    p_opt.add_argument(
        "--target-score",
        type=float,
        default=None,
        help="score<=该值且 ok 时提前结束",
    )
    p_opt.add_argument("--skip-modbus", action="store_true")
    p_opt.add_argument("--skip-capture", action="store_true")
    p_opt.add_argument("--no-wait-bias", action="store_true")
    p_opt.add_argument("--no-debug", action="store_true")
    p_opt.add_argument(
        "--no-final-score",
        action="store_true",
        help="结束后不再补一次采图+算分",
    )
    p_opt.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="某步 ok=False 仍继续（默认遇失败即停）",
    )
    p_opt.set_defaults(func=cmd_optimize)

    p_ana = sub.add_parser("analyze", help="仅分析 analysis_config 中的 TIF")
    _add_config_args(p_ana)
    p_ana.add_argument("--no-debug", action="store_true")
    p_ana.set_defaults(func=cmd_analyze)

    p_cap = sub.add_parser("capture", help="仅软触发采集 single.tif")
    _add_config_args(p_cap)
    p_cap.set_defaults(func=cmd_capture)

    p_on = sub.add_parser("modbus-on", help="连接串口并使能 ON")
    _add_config_args(p_on)
    p_on.set_defaults(func=cmd_modbus_on)

    p_off = sub.add_parser("modbus-off", help="连接串口、使能 OFF 并断开")
    _add_config_args(p_off)
    p_off.set_defaults(func=cmd_modbus_off)

    p_rd = sub.add_parser(
        "modbus-read",
        help="读五路反馈（须先 modbus-on；不切换使能）",
    )
    _add_config_args(p_rd)
    p_rd.set_defaults(func=cmd_modbus_read)

    p_set = sub.add_parser(
        "modbus-set",
        help="写五参并可选等 FIL/BIAS 稳（须先 modbus-on；不切换使能）",
    )
    _add_config_args(p_set)
    _add_param_args(p_set)
    p_set.add_argument("--wait-bias", action="store_true", help="写 BIAS 后等待平稳")
    p_set.set_defaults(func=cmd_modbus_set)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["run", *argv]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())