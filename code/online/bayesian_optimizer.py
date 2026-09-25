#!/usr/bin/env python3
"""
电子枪闭环优化系统
====================

功能流程：
    1. 写电源参数（五参：anode_kv, anode_ua, cathode_v, bias_v=200V, fil_a）
    2. 等待 BIAS 平稳
    3. 采集单帧图像
    4. 分析图像计算得分
    5. 贝叶斯优化器根据得分调整参数
    6. 循环直至收敛

依赖模块：
    - gun_modbus: 电源参数控制
    - camera_cap: 图像采集（SoftTrigger模式）
    - analysis: 图像分析与得分计算
    - scikit-optimize: 贝叶斯优化

配置文件：
    - gun_modbus_config.json: 电源Modbus配置
    - camera_cap_config.json: 相机采集配置
    - analysis_config.json: 图像分析配置

相机采集策略：
    - 在优化开始时打开相机并启动 stream
    - 每次迭代仅发送软触发采集单张图像
    - 优化结束后停止 stream 并关闭相机
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

# 设置路径
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 添加 camera_cap 路径
camera_cap_path = ROOT / "camera_cap"
if str(camera_cap_path) not in sys.path:
    sys.path.insert(0, str(camera_cap_path))

# 导入子模块
from gun_modbus.config_io import load_gun_modbus_config
from gun_modbus.params import GunParams, validate_gun_params, DEFAULT_BIAS_V
from gun_modbus.client import GunModbusClient
from camera_cap import load_config, CameraCapturer
from analysis.config_schema import load_analysis_config
from analysis.io_tif import load_tif14
from analysis.score import compute_score


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    iteration: int
    params: GunParams
    score: float
    x1: float
    x2: float
    success: bool
    reason: str = ""
    image_path: Path | None = None

    def __repr__(self):
        return (f"OptimizationResult(iter={self.iteration}, "
                f"Ua={self.params.anode_kv}kV, Ia={self.params.anode_ua}uA, "
                f"Uc={self.params.cathode_v}V, If={self.params.fil_a}A, "
                f"score={self.score:.4f}, success={self.success})")


@dataclass
class EgunOptimizer:
    """电子枪闭环优化器"""
    
    # 配置
    gun_config_path: Path = field(default=ROOT / "gun_modbus_config.json")
    camera_config_path: Path = field(default=ROOT / "camera_cap_config.json")
    analysis_config_path: Path = field(default=ROOT / "analysis_config.json")
    
    # 优化参数边界（bias_v 固定为 DEFAULT_BIAS_V）
    param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    
    # 内部状态
    _gun: GunModbusClient | None = None
    _camera: CameraCapturer | None = None
    _camera_config: Any = None
    _analysis_config: Any = None
    _results: list[OptimizationResult] = field(default_factory=list)
    
    def __enter__(self) -> EgunOptimizer:
        """上下文管理器入口：加载配置并建立连接"""
        self._load_configs()
        self._open_gun()
        self._open_camera()
        return self
    
    def __exit__(self, *args) -> None:
        """上下文管理器出口：关闭连接"""
        self._close_camera()
        self._close_gun()
    
    def _load_configs(self) -> None:
        """加载所有配置文件"""
        print("[INFO] 加载配置文件...")
        
        # 加载电源配置
        self._gun_config, _ = load_gun_modbus_config(self.gun_config_path)
        self.param_bounds = {
            'anode_kv': (self._gun_config.param_limits['anode_kv']['min'],
                        self._gun_config.param_limits['anode_kv']['max']),
            'anode_ua': (self._gun_config.param_limits['anode_ua']['min'],
                        self._gun_config.param_limits['anode_ua']['max']),
            'cathode_v': (self._gun_config.param_limits['cathode_v']['min'],
                        self._gun_config.param_limits['cathode_v']['max']),
            'fil_a': (self._gun_config.param_limits['fil_a']['min'],
                    self._gun_config.param_limits['fil_a']['max']),
        }
        
        # 加载相机配置（使用 camera_cap 模块）
        self._camera_config = load_config(str(self.camera_config_path))
        
        # 加载分析配置
        self._analysis_config, _ = load_analysis_config(self.analysis_config_path)
        
        print("[INFO] 配置加载完成")
    
    def _open_gun(self) -> None:
        """打开电源Modbus连接并使能"""
        print("[INFO] 连接电源控制器...")
        self._gun = GunModbusClient(self._gun_config)
        self._gun.open()
        self._gun.set_all_on()
        print("[INFO] 电源控制器已使能")
    
    def _close_gun(self) -> None:
        """关闭电源连接"""
        if self._gun:
            try:
                self._gun.set_all_off()
            except Exception:
                pass
            self._gun.close()
            self._gun = None
            print("[INFO] 电源连接已关闭")
    
    def _open_camera(self) -> None:
        """打开相机并启动 stream（在整个优化过程中保持打开）"""
        print("[INFO] 连接相机...")
        self._camera = CameraCapturer(config=self._camera_config)
        
        # 打开相机
        if not self._camera.open_camera():
            raise RuntimeError("无法打开相机连接")
        
        # 配置相机（软触发模式）
        if not self._camera.configure_camera():
            self._camera.close_camera()
            raise RuntimeError("无法配置相机")
        
        # 启动 stream（在优化过程中保持运行）
        if not self._camera.start_stream():
            self._camera.close_camera()
            raise RuntimeError("无法启动图像流")
        
        print("[INFO] 相机已就绪，stream 已启动")
    
    def _close_camera(self) -> None:
        """停止 stream 并关闭相机连接"""
        if self._camera:
            self._camera.close_camera()
            self._camera = None
            print("[INFO] 相机连接已关闭")
    
    def _set_params(self, params: GunParams) -> bool:
        """设置电源参数并等待BIAS平稳"""
        try:
            # 验证参数范围
            validate_gun_params(params, self._gun_config.param_limits)
            
            # 写入参数
            self._gun.set_params(params)
            print(f"[DEBUG] 设置参数: Ua={params.anode_kv}kV, Ia={params.anode_ua}uA, "
                  f"Uc={params.cathode_v}V, Ug={params.bias_v}V, If={params.fil_a}A")
            
            # 等待BIAS平稳（使用相对变化检测，不依赖目标值）
            if not self._gun.wait_bias_stable(target_v=None):
                print("[WARN] BIAS 未能在超时时间内平稳")
                return False
            
            print("[DEBUG] BIAS 已平稳")
            return True
        
        except Exception as e:
            print(f"[ERROR] 设置参数失败: {e}")
            return False
    
    def _capture_image(self) -> Path | None:
        """采集单帧图像（使用已打开的 stream）"""
        if not self._camera:
            print("[ERROR] 相机未初始化")
            return None
        
        try:
            # 获取输出路径
            output_path = ROOT / self._camera_config.get("output_dir", "captured_tif/single.tif")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 采集单张图像（复用已打开的 stream）
            result = self._camera.capture_single_image()
            if result:
                image, buffer_info = result
                
                # 保存图像
                if image.WriteTiffImage(str(output_path)):
                    print(f"[DEBUG] 图像采集完成: {output_path}")
                    return output_path
                else:
                    print(f"[ERROR] 无法保存图像到 {output_path}")
                    return None
            else:
                print("[ERROR] 图像采集失败")
                return None
        
        except Exception as e:
            print(f"[ERROR] 图像采集异常: {e}")
            return None
    
    def _analyze_image(self, image_path: Path) -> tuple[float, float, float, bool, str]:
        """分析图像并计算得分"""
        try:
            img = load_tif14(image_path)
            result = compute_score(img, config=self._analysis_config)
            
            score = float(result.score)
            x1 = float(result.x1)
            x2 = float(result.x2)
            success = result.ok
            reason = result.reason or "success"
            
            print(f"[DEBUG] 图像分析完成: score={score:.4f}, x1={x1:.4f}, x2={x2:.4f}")
            return score, x1, x2, success, reason
        
        except Exception as e:
            print(f"[ERROR] 图像分析失败: {e}")
            return float('nan'), float('nan'), float('nan'), False, str(e)
    
    def _objective_function(self, params: list[float]) -> float:
        """
        优化目标函数：给定参数，返回得分（越小越好）
        
        参数顺序：[anode_kv, anode_ua, cathode_v, fil_a]
        """
        iteration = len(self._results) + 1
        print(f"\n[ITER {iteration}] 开始优化迭代...")
        
        # 构建参数对象（bias_v 固定为 DEFAULT_BIAS_V）
        gun_params = GunParams(
            anode_kv=params[0],
            anode_ua=params[1],
            cathode_v=params[2],
            bias_v=DEFAULT_BIAS_V,
            fil_a=params[3]
        )
        
        # 步骤1: 设置参数并等待BIAS平稳
        if not self._set_params(gun_params):
            result = OptimizationResult(
                iteration=iteration,
                params=gun_params,
                score=float('inf'),
                x1=float('nan'),
                x2=float('nan'),
                success=False,
                reason="bias_not_stable"
            )
            self._results.append(result)
            return float('inf')
        
        # 步骤2: 采集图像
        image_path = self._capture_image()
        if not image_path:
            result = OptimizationResult(
                iteration=iteration,
                params=gun_params,
                score=float('inf'),
                x1=float('nan'),
                x2=float('nan'),
                success=False,
                reason="capture_failed"
            )
            self._results.append(result)
            return float('inf')
        
        # 步骤3: 分析图像计算得分
        score, x1, x2, success, reason = self._analyze_image(image_path)
        
        # 保存结果
        result = OptimizationResult(
            iteration=iteration,
            params=gun_params,
            score=score,
            x1=x1,
            x2=x2,
            success=success,
            reason=reason,
            image_path=image_path
        )
        self._results.append(result)
        
        print(f"[ITER {iteration}] 完成: score={score:.4f}")
        return score
    
    def optimize(self, max_iterations: int = 30, n_initial_points: int = 5) -> OptimizationResult:
        """
        执行贝叶斯优化
        
        Args:
            max_iterations: 最大迭代次数
            n_initial_points: 初始随机采样点数
        
        Returns:
            最优结果
        """
        print(f"\n[INFO] 开始贝叶斯优化，最大迭代次数: {max_iterations}")
        print(f"[INFO] 参数范围: {self.param_bounds}")
        print(f"[INFO] BIAS_V 固定为: {DEFAULT_BIAS_V}V")
        
        # 定义优化空间
        search_space = [
            Real(*self.param_bounds['anode_kv'], name='anode_kv'),
            Real(*self.param_bounds['anode_ua'], name='anode_ua'),
            Real(*self.param_bounds['cathode_v'], name='cathode_v'),
            Real(*self.param_bounds['fil_a'], name='fil_a'),
        ]
        
        # 执行贝叶斯优化
        result = gp_minimize(
            func=self._objective_function,
            dimensions=search_space,
            n_calls=max_iterations,
            n_initial_points=n_initial_points,
            random_state=42,
            verbose=True
        )
        
        # 提取最优结果
        best_idx = np.argmin([r.score for r in self._results if r.success])
        best_result = self._results[best_idx]
        
        print("\n" + "="*60)
        print("[RESULT] 优化完成")
        print(f"最优得分: {best_result.score:.4f}")
        print(f"最优参数: Ua={best_result.params.anode_kv}kV, "
              f"Ia={best_result.params.anode_ua}uA, "
              f"Uc={best_result.params.cathode_v}V, "
              f"If={best_result.params.fil_a}A")
        print("="*60)
        
        return best_result
    
    def get_results(self) -> list[OptimizationResult]:
        """获取所有迭代结果"""
        return self._results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='电子枪闭环贝叶斯优化')
    parser.add_argument('--max-iter', type=int, default=30, help='最大迭代次数')
    parser.add_argument('--init-points', type=int, default=5, help='初始采样点数')
    parser.add_argument('--gun-config', type=Path, default=ROOT / 'gun_modbus_config.json',
                        help='电源配置文件路径')
    parser.add_argument('--camera-config', type=Path, default=ROOT / 'camera_cap_config.json',
                        help='相机配置文件路径')
    parser.add_argument('--analysis-config', type=Path, default=ROOT / 'analysis_config.json',
                        help='分析配置文件路径')
    
    args = parser.parse_args()
    
    try:
        with EgunOptimizer(
            gun_config_path=args.gun_config,
            camera_config_path=args.camera_config,
            analysis_config_path=args.analysis_config
        ) as optimizer:
            best_result = optimizer.optimize(
                max_iterations=args.max_iter,
                n_initial_points=args.init_points
            )
            
            # 保存优化历史
            history_path = ROOT / 'optimization_history.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                history_data = []
                for r in optimizer.get_results():
                    history_data.append({
                        'iteration': r.iteration,
                        'params': {
                            'anode_kv': r.params.anode_kv,
                            'anode_ua': r.params.anode_ua,
                            'cathode_v': r.params.cathode_v,
                            'bias_v': r.params.bias_v,
                            'fil_a': r.params.fil_a
                        },
                        'score': r.score,
                        'x1': r.x1,
                        'x2': r.x2,
                        'success': r.success,
                        'reason': r.reason,
                        'image_path': str(r.image_path) if r.image_path else None
                    })
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n[INFO] 优化历史已保存至: {history_path}")
            
    except Exception as e:
        print(f"[FATAL] 优化过程发生错误: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())