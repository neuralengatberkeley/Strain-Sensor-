import PySpin
import numpy as np
import cv2

def acquire_and_display(serial):
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = None

    try:
        # Get camera by serial number
        cam = cam_list.GetBySerial(serial)
        cam.Init()

        # Set trigger mode OFF
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        cam.BeginAcquisition()

        print("Streaming... Press 'q' to quit.")
        while True:
            img = cam.GetNextImage(1000)

            if img.IsIncomplete():
                print("Incomplete image:", img.GetImageStatus())
                img.Release()
                continue

            # Image conversion
            width = img.GetWidth()
            height = img.GetHeight()
            img_data = img.GetData()
            img_np = np.array(img_data, dtype=np.uint8).reshape((height, width))

            cv2.imshow("Live View", img_np)
            img.Release()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.EndAcquisition()
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print("PySpin error:", ex)

    finally:
        if cam is not None:
            del cam  # Ensure camera object is gone
        cam_list.Clear()
        del cam_list       # Fully remove camera list
        system.ReleaseInstance()
        del system         # Fully remove system object
        cv2.destroyAllWindows()

# Run it
acquire_and_display('25183199')
