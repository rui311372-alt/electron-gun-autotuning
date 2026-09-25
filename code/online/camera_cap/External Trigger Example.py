import logging
import sys
import os
os.add_dll_directory(r"D:\Example_Code_Python\Pleora Examples\SDK\dll\x64\Release")

import SLDevicePythonWrapper
from SLDevicePythonWrapper import DeviceInterface,SLDevice,SLError,ExposureModes,SLImage

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
    
    ExternalTriggerExample(device)
    
    err = device.CloseCamera()
    if err != SLError.SL_ERROR_SUCCESS:
        logging.error(f"Failed to CloseCamera with error: {err}")
        return -2
    
    logging.info("Successfully closed camera")
    
    return 0

def ExternalTriggerExample(device: SLDevice) -> None:
    numFrames = 20
    dds = False
    timeout = 10 * 1000
    receivedFrames = 0
    exposureMode = ExposureModes.trig_mode
   
    logging.info("Initiialising External Trigger")

      # Configure the device
    err = device.SetExposureMode(exposureMode)
    if err != SLError.SL_ERROR_SUCCESS: 
        logging.error(f"Failed to set exposure mode to {exposureMode} with error: {err}")
        return
    
    logging.info(f"Set exposure mode to {exposureMode}")
    
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
    
    logging.info("Started Stream") 
    logging.info("Awaiting hardware triggers")
    
    timedOut = False # Whether acquire image timed out

    # Captures up to numFrames frames
    # AcquireImage will block the thread until a frame is received or it times out
    # Frame rate (as well as exposure time) will depend on trigger frequency
    # For high frame rates, images should not be saved in loop as this can slow AcquisitionExample
    while receivedFrames < numFrames and not timedOut:
        bufferInfo = device.AcquireImage(image, timeout=timeout)
        filename = f"{imageSaveDirectory}ExternalTriggerCapture{bufferInfo.frameCount}.tif"

        if bufferInfo.error == SLError.SL_ERROR_SUCCESS:                # Frame acquired successfully
            receivedFrames += 1
            logging.info(f"Received frame #{receivedFrames} ({bufferInfo.frameCount}) with dims: {bufferInfo.width}x{bufferInfo.height}")
            if image.WriteTiffImage(filename) is False:
                logging.error("Failed to save image")
        elif bufferInfo.error == SLError.SL_ERROR_MISSING_PACKETS:      # Frame acquired with missing packets
            receivedFrames += 1
            logging.info(f"Received frame #{receivedFrames} ({bufferInfo.frameCount}) with dims: {bufferInfo.width}x{bufferInfo.height}, missing packets: {bufferInfo.missingPackets}")
            if image.WriteTiffImage(filename) is False:
                logging.error(": Failed to save image")
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
    
    logging.info("Stopped Stream")

if __name__ == "__main__":
    sys.exit(main())