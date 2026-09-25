from SLDevicePythonWrapper import *
import time
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

imageSaveDirectory = "C:\\SLDevice\\Examples\\captured_images\\"
deviceInterface = DeviceInterface.PLEORA

def main() -> int:
    # Open a connection to a device with a specified interface
    device = SLDevice(deviceInterface)
    
    err = device.OpenCamera()
    if err != SLError.SL_ERROR_SUCCESS: 
        logging.error(f"Failed to Open Camera with error: {err}")
        return -1
    
    logging.info("Successfully opened camera")
    
    StartStreamWithCallbackExample(device)
    
    err = device.CloseCamera()
    if err != SLError.SL_ERROR_SUCCESS:
        logging.error(f"Failed to CloseCamera with error: {err}")
        return -2
    
    logging.info("Successfully closed camera")
    
    return 0

class CallbackCounter:
    def __init__(self):
        self.value = 0

def callback_fn(view: memoryview, bufferInfo: SLBufferInfo, counter: CallbackCounter) -> None:
      filename = f"{imageSaveDirectory}StreamCallbackCapture{bufferInfo.frameCount}.tif"
      if bufferInfo.error == SLError.SL_ERROR_SUCCESS:                  # Frame acquired successfully
            counter.value += 1
            logging.info(f"Received frame #{counter.value} ({bufferInfo.frameCount}) with dims: {bufferInfo.width}x{bufferInfo.height}")
            image = SLImage.Array2Frame(np.frombuffer(view, dtype=np.uint16).reshape(bufferInfo.height, bufferInfo.width))
            image.WriteTiffImage(filename)
      elif bufferInfo.error == SLError.SL_ERROR_MISSING_PACKETS:        # Frame acquired with missing packets
            counter.value += 1
            logging.info(f"Received frame #{counter.value} ({bufferInfo.frameCount}) with dims: {bufferInfo.width}x{bufferInfo.height}, missing packets: {bufferInfo.missingPackets}")
            image = SLImage.Array2Frame(np.frombuffer(view, dtype=np.uint16).reshape(bufferInfo.height, bufferInfo.width))
            image.WriteTiffImage(filename)
      else:
            logging.error(f"Received error in callback: {bufferInfo.error}")

def StartStreamWithCallbackExample(device: SLDevice) -> None:
      secondsToStream = 5
      expTime = 100
      dds = False
      exposureMode = ExposureModes.xfps_mode
      
      logging.info("Initialising Start Stream With Callback")

      # Configure the device            
      err = device.SetExposureMode(exposureMode)
      if err != SLError.SL_ERROR_SUCCESS: 
            logging.error(f"Failed to set exposure mode to {exposureMode} with error: {err}")
            return
      
      logging.info(f"Set exposure mode to {exposureMode}")
      
      err = device.SetExposureTime(expTime)
      if err != SLError.SL_ERROR_SUCCESS: 
            logging.error(f"Failed to set exposure time to {expTime} with error: {err}")
            return
      
      logging.info(f"Set exposure time to {expTime}")
      
      err = device.SetDDS(dds)
      if err != SLError.SL_ERROR_SUCCESS: 
            logging.error(f"Failed to set DDS to {dds} with error: {err}")
            return
      
      logging.info(f"Set DDS to {dds}")

      counter = CallbackCounter()

      # Start Stream
      err = device.StartStream(callback=callback_fn, counter=counter)
      if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to start stream with error: {err}")
            return
      
      logging.info("Started stream") 
        
      startTime = time.time()
      
      # Continuously acquires images until secondsToStream elapses
      # The callback is automatically invoked after each new frame is received
      # For high frame rates, images should not be saved directly from the callback as this can slow AcquisitionExample
      while time.time() - startTime < secondsToStream:
            time.sleep(expTime / 1000)
                 
      # Stop Stream
      err = device.StopStream()
      if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to stop stream with error: {err}")
            return
      
      logging.info("Stopped stream")
                  
if __name__ == "__main__":
    sys.exit(main())