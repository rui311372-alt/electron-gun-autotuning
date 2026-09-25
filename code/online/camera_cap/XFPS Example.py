from SLDevicePythonWrapper import *
import time
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
    
    XFPSExample(device)
    
    err = device.CloseCamera()
    if err != SLError.SL_ERROR_SUCCESS:
        logging.error(f"Failed to CloseCamera with error: {err}")
        return -2
    
    logging.info("Successfully closed camera")
    
    return 0

def XFPSExample(device: SLDevice) -> None:
      expTime = 100
      dds = False
      secondsToStream = 5
      exposureMode = ExposureModes.xfps_mode
      
      logging.info("Initialising xfps mode")
      
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

      # Build SLImage object to read frames into
      image = SLImage(device.GetImageXDim(), device.GetImageYDim())
      bufferInfo: SLBufferInfo = None

      # Start Stream
      err = device.StartStream()
      if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to StartStream with error: {err}")
            return
      
      logging.info("Started stream") 
        
      startTime = time.time()
      
      # Continuously acquires images until secondsToStream elapses
      # AcquireImage will block the thread until a frame is received or it times out
      # For high frame rates, images should not be saved in loop as this can slow AcquisitionExample
      while time.time() - startTime < secondsToStream:
            bufferInfo = device.AcquireImage(image)
            readTime = time.time() - startTime
            filename = f"{imageSaveDirectory}XFPSCapture{bufferInfo.frameCount}.tif"
            
            if bufferInfo.error == SLError.SL_ERROR_SUCCESS:                # Frame acquired successfully
                  logging.info(f"Received frame #{bufferInfo.frameCount} with dims: {bufferInfo.width}x{bufferInfo.height} at {readTime}")
                  if image.WriteTiffImage(filename) is False:
                        logging.error("Failed to save image")
            elif bufferInfo.error == SLError.SL_ERROR_MISSING_PACKETS:      # Frame acquired with missing packets
                  logging.info(f"Received frame #{bufferInfo.frameCount} with dims: {bufferInfo.width}x{bufferInfo.height} at {readTime}, missing packets: {bufferInfo.missingPackets}")
                  if image.WriteTiffImage(filename) is False:
                        logging.error("Failed to save image")
            elif bufferInfo.error == SLError.SL_ERROR_TIMEOUT:
                  logging.warning("Timed out waiting for frame")
            else:
                  logging.error(f"Failed to acquire image with error: {bufferInfo.error}")
                            
      # Stop Stream
      err = device.StopStream()
      if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to stop stream with error: {err}")
            return
      
      logging.info("Stopped stream")
      
if __name__ == "__main__":
    sys.exit(main())