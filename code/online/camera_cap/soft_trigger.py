from pathlib import Path
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_dir, "SDK", "dll", "x64", "Release")
os.add_dll_directory(dll_path)

import SLDevicePythonWrapper
from SLDevicePythonWrapper import DeviceInterface, SLDevice, SLError, ExposureModes, SLImage, SLBufferInfo


class CameraRuntimeError(Exception):
    pass


def _read_exposure_time_10us(device):
    """Best-effort wrapper for the SDK's ``GetExposureTime(int&)`` binding.

    The Python binding normally returns ``(SLError, value_10us)`` for a C++
    output parameter.  Keep the capture path usable if an older wrapper does
    not expose that overload.
    """
    try:
        result = device.GetExposureTime()
    except Exception as exc:
        return None, f"readback_unavailable:{exc}"
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        status, value = result[0], result[1]
        if status != SLError.SL_ERROR_SUCCESS:
            return None, f"readback_error:{status}"
        try:
            return int(value), None
        except (TypeError, ValueError):
            return None, f"readback_invalid:{value!r}"
    # Some wrapper builds return only the value.  A positive value is a valid
    # exposure in 10-us units; zero is ambiguous and is not reported as data.
    if isinstance(result, (int, float)) and result > 0:
        return int(result), None
    return None, f"readback_unrecognized:{result!r}"


class SoftTriggerSession:
    def __init__(self, config):
        self.config = config
        self.device = None
        self.image = None
        self.is_streaming = False

    def __enter__(self):
        self.device = SLDevice(DeviceInterface.PLEORA)
        err = self.device.OpenCamera()
        if err != SLError.SL_ERROR_SUCCESS:
            raise CameraRuntimeError(f"Failed to open camera: {err}")

        err = self.device.SetExposureMode(ExposureModes.trig_mode)
        if err != SLError.SL_ERROR_SUCCESS:
            raise CameraRuntimeError(f"Failed to set exposure mode: {err}")

        # The optimizer writes this field before capture.  Set it after the
        # mode switch because the SDK resets exposure when trig_mode changes.
        exposure_time_us = getattr(self.config, "exposure_time_us", None)
        if exposure_time_us is not None:
            exposure_time_us = int(exposure_time_us)
            if exposure_time_us <= 0:
                raise CameraRuntimeError("exposure_time_us must be positive")
            err = self.device.SetExposureTime(exposure_time_us)
            if err != SLError.SL_ERROR_SUCCESS:
                raise CameraRuntimeError(
                    f"Failed to set exposure time to {exposure_time_us} ms: {err}"
                )
            readback_10us, readback_note = _read_exposure_time_10us(self.device)
            if readback_10us is None:
                print(
                    "camera exposure: "
                    f"requested={exposure_time_us} ms, readback=- ({readback_note})"
                )
            else:
                print(
                    "camera exposure: "
                    f"requested={exposure_time_us} ms, "
                    f"readback={readback_10us} x 10us = {readback_10us / 100:.3f} ms"
                )

        self.image = SLImage(self.device.GetImageXDim(), self.device.GetImageYDim())

        err = self.device.StartStream()
        if err != SLError.SL_ERROR_SUCCESS:
            raise CameraRuntimeError(f"Failed to start stream: {err}")

        self.is_streaming = True
        import time
        sleep_time = self.config.start_sleep_seconds if hasattr(self.config, "start_sleep_seconds") else 5
        time.sleep(sleep_time)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_streaming:
            self.device.StopStream()
        if self.device:
            self.device.CloseCamera()

    def grab_to_tif(self, tif_out: Path = None) -> Path:
        if tif_out is None:
            tif_out = Path(self.config.output_dir) if hasattr(self.config, "output_dir") else Path("captured_tif/single.tif")

        tif_out.parent.mkdir(parents=True, exist_ok=True)
        err = self.device.SoftwareTrigger()
        if err != SLError.SL_ERROR_SUCCESS:
            raise CameraRuntimeError(f"Failed to send software trigger: {err}")

        buffer_info = self.device.AcquireImage(self.image)
        if buffer_info.error != SLError.SL_ERROR_SUCCESS:
            raise CameraRuntimeError(f"Failed to acquire image: {buffer_info.error}")

        if not self.image.WriteTiffImage(str(tif_out)):
            raise CameraRuntimeError(f"Failed to save image to {tif_out}")
        return tif_out
