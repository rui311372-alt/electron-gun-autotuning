from SLDevicePythonWrapper import *
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
    
    SequenceExample(device)
    
    err = device.CloseCamera()
    if err != SLError.SL_ERROR_SUCCESS:
        logging.error(f"Failed to CloseCamera with error: {err}")
        return -2
    
    logging.info("Successfully closed camera")
    
    return 0

def SequenceExample(device: SLDevice) -> None:
      numFrames = 20
      expTime = 100
      timeout = 10 * expTime + 1000 
      dds = False
      exposureMode = ExposureModes.seq_mode
      receivedFrames = 0

      logging.info("Initialising Sequence")

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
      
      err = device.SetNumberOfFrames(numFrames)
      if err != SLError.SL_ERROR_SUCCESS: 
            logging.error(f"Failed to set number of frames to {numFrames} with error: {err}")
            return
      
      logging.info(f"Set number of frames to {numFrames}")
      
      err = device.SetDDS(dds)
      if err != SLError.SL_ERROR_SUCCESS: 
            logging.error(f"Failed to set DDS to {dds} with error: {err}")
            return
      
      logging.info(f"Set DDS to {dds}")
      
      # Build SLImage object to read frames into
      image = SLImage(device.GetImageXDim(), device.GetImageYDim(), numFrames)      
      bufferInfo: SLBufferInfo = None

      # Start Stream
      err = device.StartStream()
      if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to StartStream with error: {err}")
            return
      
      logging.info("Started stream")    
      
      # Send a software trigger to start sequence capture
      err = device.SoftwareTrigger()
      if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to send software trigger with error: {err}")
            return
      
      logging.info("Sent software trigger")
      
      timedOut = False # Whether AcquireImage timed out
      
      # Captures a sequence of numFrames frames
      # If a frame is dropped, it will eventually time out
      # AcquireImage will block the thread until a frame is received or it times out
      while receivedFrames < numFrames and not timedOut:
            bufferInfo = device.AcquireImage(image, frame=receivedFrames, timeout=timeout)
            
            if bufferInfo.error == SLError.SL_ERROR_SUCCESS:                # Frame acquired successfully
                  receivedFrames += 1
                  logging.info(f"Received frame #{receivedFrames} ({bufferInfo.frameCount}) with dims: {bufferInfo.width}x{bufferInfo.height}")
            elif bufferInfo.error == SLError.SL_ERROR_MISSING_PACKETS:      # Frame acquired with missing packets
                  receivedFrames += 1
                  logging.info(f"Received frame #{receivedFrames} ({bufferInfo.frameCount}) with dims: {bufferInfo.width}x{bufferInfo.height}, missing packets: {bufferInfo.missingPackets}")
            elif bufferInfo.error == SLError.SL_ERROR_TIMEOUT:
                  logging.warning("Timed out waiting for frame")  
                  timedOut = True
            else:
                  logging.error(f"Failed to acquire image with error: {bufferInfo.error}")
                                   
      # Stop Stream
      err = device.StopStream()
      if err != SLError.SL_ERROR_SUCCESS:
            logging.error(f"Failed to stop stream with error: {err}")
            return
      
      logging.info("Stopped stream")
      
      framesToCrop = numFrames - receivedFrames
      if framesToCrop > 0:
          image.DeleteLastNSlices(framesToCrop) # If all frames were not acquired, trim down the image
          logging.info(f"Cropped last {framesToCrop}")
          
      # Don't bother saving an image if it has no frames
      if image.GetDepth() == 0:
            return
      
      # Save the captured images
      filename = f"{imageSaveDirectory}SequenceCapture.tif"
      if image.WriteTiffImage(filename, 16):
            logging.info(f"Saved image as {filename}")
      else:
            logging.error("Error saving image.")

if __name__ == "__main__":
    sys.exit(main())