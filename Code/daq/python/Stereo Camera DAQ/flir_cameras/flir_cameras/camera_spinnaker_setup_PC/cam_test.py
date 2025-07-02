import PySpin
system = PySpin.System.GetInstance()

serial_1 = '22157398'
cam_list = system.GetCameras()
cam_1 = cam_list.GetBySerial(serial_1)
cam_1.Init()


cam_1.TriggerMode.SetValue(PySpin.TriggerMode_Off)
cam_1.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
cam_1.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
cam_1.TriggerMode.SetValue(PySpin.TriggerMode_On)

cam_1.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
cam_1.BeginAcquisition()

im = cam_1.GetNextImage()

width = im.GetWidth()
height = im.GetHeight()

## Make a texture
im_data = im.GetData()
