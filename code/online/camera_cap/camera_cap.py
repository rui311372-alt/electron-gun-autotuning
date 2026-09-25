import logging
import sys
import os
import time
import json
from typing import Optional, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # 项目根目录
dll_path = os.path.join(script_dir, "SDK", "dll", "x64", "Release")
os.add_dll_directory(dll_path)

# 默认配置
DEFAULT_CONFIG = {
    "output_dir": "captured_tif/single.tif",
    "start_sleep_seconds": 5,
    "exposure_time_us": 1000
}


def load_config(config_path: str = None) -> dict:
    """
    加载相机采集配置文件

    Args:
        config_path: 配置文件路径，默认为项目根目录下的 camera_cap_config.json

    Returns:
        dict: 配置字典
    """
    if config_path is None:
        config_path = os.path.join(project_root, "camera_cap_config.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 合并默认配置，确保所有字段都存在
            return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            logging.error(f"Failed to load config file {config_path}: {e}")
            return DEFAULT_CONFIG
    else:
        logging.warning(f"Config file {config_path} not found, using default config")
        return DEFAULT_CONFIG


import SLDevicePythonWrapper
from SLDevicePythonWrapper import DeviceInterface, SLDevice, SLError, ExposureModes, SLImage, SLBufferInfo

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')


class CameraCapturer:
    """相机采集器类，提供单张图像采集功能"""

    def __init__(self, device_interface: DeviceInterface = DeviceInterface.PLEORA, config: dict = None):
        self.device_interface = device_interface
        self.device = None
        self.image = None
        self.is_streaming = False
        self.config = config if config else load_config()

    def open_camera(self) -> bool:
        """打开相机连接"""
        try:
            self.device = SLDevice(self.device_interface)
            err = self.device.OpenCamera()
            if err != SLError.SL_ERROR_SUCCESS:
                logging.error(f"Failed to Open Camera with error: {err}")
                return False

            logging.info("Successfully opened camera")
            return True
        except Exception as e:
            logging.error(f"Error opening camera: {e}")
            return False

    def configure_camera(self, exposure_mode: ExposureModes = ExposureModes.trig_mode, dds: bool = False) -> bool:
        """配置相机参数"""
        if not self.device:
            logging.error("Camera not opened")
            return False

        err = self.device.SetExposureMode(exposure_mode)
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to set exposure mode to {exposure_mode} with error: {err}")
            return False

        logging.info(f"Set exposure mode to {exposure_mode}")

        exposure_time_us = int(self.config.get("exposure_time_us", 1000))
        err = self.device.SetExposureTime(exposure_time_us)
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to set exposure time to {exposure_time_us}us with error: {err}")
            return False

        logging.info(f"Set exposure time to {exposure_time_us}us")

        err = self.device.SetDDS(dds)
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to set DDS to {dds} with error: {err}")
            return False

        logging.info(f"Set DDS to {dds}")

        # 创建图像对象
        self.image = SLImage(self.device.GetImageXDim(), self.device.GetImageYDim())
        return True

    def start_stream(self) -> bool:
        """开始图像流"""
        if not self.device:
            logging.error("Camera not opened")
            return False

        err = self.device.StartStream()
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to start stream with error: {err}")
            return False

        self.is_streaming = True
        # 等待曝光完成，否则图片比较黑（从配置读取等待时长）
        sleep_time = self.config.get("start_sleep_seconds", 5)
        time.sleep(sleep_time)
        logging.info(f"Started stream (waited {sleep_time}s)")
        return True

    def capture_single_image(self) -> Optional[Tuple[SLImage, SLBufferInfo]]:
        """
        采集单张图像

        Returns:
            Optional[Tuple[SLImage, SLBufferInfo]]: 图像对象和缓冲区信息，如果采集失败返回None
        """
        if not self.device or not self.image:
            logging.error("Camera not initialized")
            return None

        if not self.is_streaming:
            logging.error("Stream not started")
            return None

        # 发送软件触发
        err = self.device.SoftwareTrigger()
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to send software trigger with error: {err}")
            return None

        logging.info("Sent software trigger")

        # 获取图像
        buffer_info = self.device.AcquireImage(self.image)

        if buffer_info.error == SLError.SL_ERROR_SUCCESS:
            logging.info(
                f"Acquired frame #{buffer_info.frameCount} with dims: {buffer_info.width}x{buffer_info.height}")
            return self.image, buffer_info
        elif buffer_info.error == SLError.SL_ERROR_MISSING_PACKETS:
            logging.warning(
                f"Acquired frame #{buffer_info.frameCount} with missing packets: {buffer_info.missingPackets}")
            return self.image, buffer_info
        elif buffer_info.error == SLError.SL_ERROR_TIMEOUT:
            logging.warning("Timed out whilst waiting for frame")
            return None
        else:
            logging.error(f"Failed to acquire image with error: {buffer_info.error}")
            return None

    def stop_stream(self) -> bool:
        """停止图像流"""
        if not self.device:
            logging.error("Camera not opened")
            return False

        err = self.device.StopStream()
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to stop stream with error: {err}")
            return False

        self.is_streaming = False
        logging.info("Stopped stream")
        return True

    def close_camera(self) -> bool:
        """关闭相机连接"""
        if not self.device:
            logging.error("Camera not opened")
            return False

        # 如果正在流，先停止
        if self.is_streaming:
            self.stop_stream()

        err = self.device.CloseCamera()
        if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to CloseCamera with error: {err}")
            return False

        logging.info("Successfully closed camera")
        return True


def capture_single_frame(save_path: Optional[str] = None) -> bool:
    """
    单次采集一张图像的便捷函数

    Args:
        save_path: 可选的保存路径，如果提供则保存为TIFF文件

    Returns:
        bool: 采集是否成功
    """
    capturer = CameraCapturer()

    try:
        if not capturer.open_camera():
            return False

        if not capturer.configure_camera():
            return False

        if not capturer.start_stream():
            return False

        result = capturer.capture_single_image()
        if result:
            image, buffer_info = result

            if save_path:
                if image.WriteTiffImage(save_path) is False:
                    logging.error(f"Failed to save image to {save_path}")
                    return False
                logging.info(f"Image saved to {save_path}")

            return True

        return False

    finally:
        capturer.close_camera()


def main():
    """测试函数：采集一张图像并保存"""
    logging.info("Testing camera capture...")

    # 加载配置
    config = load_config()

    # 获取输出文件路径（从配置读取）
    output_path = config.get("output_dir", "captured_tif/single.tif")
    save_path = os.path.join(project_root, output_path)

    # 确保输出目录存在
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # 采集单张图像
    success = capture_single_frame(save_path)

    if success:
        logging.info("Test capture completed successfully")
        return 0
    else:
        logging.error("Test capture failed")
        return -1


if __name__ == "__main__":
    sys.exit(main())
