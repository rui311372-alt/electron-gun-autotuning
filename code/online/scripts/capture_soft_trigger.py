"""命令行：软触发采集一帧或多帧 TIF（与 Pleora Examples 完全一致）。"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 关键：添加 Pleora Examples 目录到 sys.path（与示例脚本完全一致）
# 这样会优先导入该目录下的 SLDevicePythonWrapper.pyd
PLEORA_EXAMPLES_DIR = str(ROOT / "Pleora Examples")
sys.path.insert(0, PLEORA_EXAMPLES_DIR)

# 添加 DLL 目录（用于依赖的 .dll 文件）
# os.add_dll_directory(PLEORA_SDK_DLL_DIR)
os.add_dll_directory(r"D:\Example_Code_Python\Pleora Examples\SDK\dll\x64\Release")

import SLDevicePythonWrapper

# 导入 SDK 模块（与 Pleora Examples 完全相同的导入方式）
try:
    from SLDevicePythonWrapper import DeviceInterface, SLDevice, SLError, ExposureModes, SLImage
    # 打印实际导入的模块路径（用于调试）
    # print(f"导入的模块路径: {SLDevicePythonWrapper.__file__}")
except ImportError as exc:
    print(f"无法 import SLDevicePythonWrapper。请检查路径: {PLEORA_EXAMPLES_DIR}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="平板探测器软触发采集 TIF（与 Pleora Examples 一致）")
    parser.add_argument("-n", "--count", type=int, default=1, help="连续采集帧数")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="每帧前等待回车，便于手动对齐工况（默认不等待）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 输出目录
    out_base = (ROOT / "captured_tif").resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    # 与 Pleora Examples 完全一致的初始化流程
    device = SLDevice(DeviceInterface.PLEORA)

    err = device.OpenCamera()
    if err != SLError.SL_ERROR_SUCCESS:
        logging.error(f"Failed to Open Camera with error: {err}")
        return -1
    logging.info("Successfully opened camera")

    # 配置参数（与 Pleora Examples 一致）
    dds = False
    exposureMode = ExposureModes.trig_mode

    err = device.SetExposureMode(exposureMode)
    if err != SLError.SL_ERROR_SUCCESS:
        logging.error(f"Failed to set exposure mode to {exposureMode} with error: {err}")
        device.CloseCamera()
        return -1
    logging.info(f"Set exposure mode to {exposureMode}")

    err = device.SetDDS(dds)
    if err != SLError.SL_ERROR_SUCCESS:
        logging.error(f"Failed to set DDS to {dds} with error: {err}")
        device.CloseCamera()
        return -1
    logging.info(f"Set DDS to {dds}")

    # 创建图像对象
    image = SLImage(device.GetImageXDim(), device.GetImageYDim())

    # 开始流
    err = device.StartStream()
    if err != SLError.SL_ERROR_SUCCESS:
        logging.error(f"Failed to start stream with error: {err}")
        device.CloseCamera()
        return -1
    logging.info("Started stream")

    # 采集图像
    try:
        for i in range(args.count):
            if args.interactive:
                input(f"[{i + 1}/{args.count}] 回车发送软触发… ")

            # 发送软触发
            err = device.SoftwareTrigger()
            if err != SLError.SL_ERROR_SUCCESS:
                logging.error(f"Failed to send software trigger with error: {err}")
                break
            logging.info("Sent software trigger")

            # 采集图像
            bufferInfo = device.AcquireImage(image)

            # 构建输出路径
            if args.count == 1:
                filename = str(out_base / "single.tif")
            else:
                filename = str(out_base / f"single_{i + 1:03d}.tif")

            # 处理采集结果（与 Pleora Examples 一致）
            if bufferInfo.error == SLError.SL_ERROR_SUCCESS:
                logging.info(f"Read new frame #{bufferInfo.frameCount} with dims: {bufferInfo.width}x{bufferInfo.height}")
                if image.WriteTiffImage(filename) is False:
                    logging.error("Failed to save image")
            elif bufferInfo.error == SLError.SL_ERROR_MISSING_PACKETS:
                logging.info(f"Read new frame #{bufferInfo.frameCount} with dims: {bufferInfo.width}x{bufferInfo.height}, missing packets: {bufferInfo.missingPackets}")
                if image.WriteTiffImage(filename) is False:
                    logging.error("Failed to save image")
            elif bufferInfo.error == SLError.SL_ERROR_TIMEOUT:
                logging.warning("Timed out whilst waiting for frame")
            else:
                logging.error(f"Failed to acquire image with error: {bufferInfo.error}")

            print(filename)

    except KeyboardInterrupt:
        logging.info("Interrupted")
    finally:
        # 停止流
        err = device.StopStream()
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to stop stream with error: {err}")
        logging.info("Stopped stream")

        # 关闭相机
        err = device.CloseCamera()
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to CloseCamera with error: {err}")
        logging.info("Successfully closed camera")

    return 0


if __name__ == "__main__":
    sys.exit(main())
