from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import StringProperty

from datetime import datetime
import os, time
import scipy.io as sio
import numpy as np

import PySpin
system = PySpin.System.GetInstance()

class ImageEventHandler(PySpin.ImageEventHandler):
    def __init__(self, cam, folder_path, cam_stem):
        super(ImageEventHandler, self).__init__()
        self._image_count = 0
        self.folder_path = folder_path
        self.cam_stem = cam_stem
        self.i1 = []; self.t1 = []; self.id1 = [];
        self.T = []; self.T0 = time.time()
        self.latest_image = None
        del cam

    def OnImageEvent(self, image):
        if image.IsIncomplete():
            print('Image incomplete with image status %i...' % image.GetImageStatus())
        else:
            cd1 = image.GetChunkData()
            self.t1.append(cd1.GetTimestamp())
            self.id1.append(image.GetFrameID())
            zz = str(self._image_count).zfill(6)
            image.Save(os.path.join(self.folder_path, f'{self.cam_stem}_{zz}.raw'))
            self.T.append(time.time() - self.T0)
            self._image_count += 1
            self.latest_image = image

    def get_image_count(self):
        return self._image_count

class CameraApp(App):
    def load(self, *args):
        now = datetime.now()
        folder_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.folder_path = os.path.join("C:/", folder_str)
        self.folder_label.text = self.folder_path
        os.mkdir(self.folder_path)
        self.state = 'loaded'

    def start(self, *args):
        if self.state == 'loaded':
            serial_1 = '25183199'
            serial_2 = '25185174'
            cam_list = system.GetCameras()
            num_cams = cam_list.GetSize()
            print(f"Number of cameras detected: {num_cams}")

            self.cam_1 = cam_list.GetBySerial(serial_1)
            self.cam_2 = cam_list.GetBySerial(serial_2)
            print(self.cam_1)
            print(self.cam_2)

            self.cam_1.Init()
            self.cam_2.Init()
            print("Init Success")


            self.cam_1.LineSelector.SetValue(PySpin.LineSelector_Line2)
            self.cam_2.LineSelector.SetValue(PySpin.LineSelector_Line2)
            self.cam_1.LineMode.SetValue(PySpin.LineMode_Input)

            self.cam_2.LineMode.SetValue(PySpin.LineMode_Input)

            self.cam_2.TriggerSource.SetValue(PySpin.TriggerSource_Line2)
            self.cam_2.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
            self.cam_2.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
            self.cam_2.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)

            self.cam_1.TriggerSource.SetValue(PySpin.TriggerSource_Line2)
            self.cam_1.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            self.cam_1.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)

            self.cam_1.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)


            self.cam_1.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.cam_2.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

            self.cam_2.TriggerMode.SetValue(PySpin.TriggerMode_On)
            self.cam_1.TriggerMode.SetValue(PySpin.TriggerMode_On)

            time.sleep(2)

            start_time = time.time()
            while time.time() - start_time < 3.0:
                current_status = self.cam_1.LineStatus.GetValue()
                print(f"Line2: {'HIGH' if current_status else 'LOW'} at {time.time() - start_time:.3f}s")
                time.sleep(0.01)

            self.image_event_handler1 = ImageEventHandler(self.cam_1, self.folder_path, 'cam_1')
            self.image_event_handler2 = ImageEventHandler(self.cam_2, self.folder_path, 'cam_2')
            self.cam_1.RegisterEventHandler(self.image_event_handler1)
            self.cam_2.RegisterEventHandler(self.image_event_handler2)

            self.state = 'started'
            print('started')

            self.cam_2.BeginAcquisition()
            self.cam_1.BeginAcquisition()



            self.wait_for_images(self.image_event_handler1)
            self.wait_for_images(self.image_event_handler2)


        else:
            print('State is %s, not loaded' % self.state)

    def wait_for_images(self, handler):
        while self.state == 'started':
            pass

    def stop(self, *args):
        self.state = 'stop'
        self.cam_1.EndAcquisition()
        self.cam_2.EndAcquisition()

        # Print image counts from both cameras
        print(f"Final image count - Cam 1 ({self.image_event_handler1.cam_stem}): {self.image_event_handler1._image_count}")
        print(f"Final image count - Cam 2 ({self.image_event_handler2.cam_stem}): {self.image_event_handler2._image_count}")

        d = dict(t1=self.image_event_handler1.t1, t2=self.image_event_handler2.t1,
                 t1_backup=self.image_event_handler1.T, t2_backup=self.image_event_handler2.T,
                 f1=self.image_event_handler1.id1, f2=self.image_event_handler2.id1)

        sio.savemat(os.path.join(self.folder_path, "meta_data.mat"), d)
        print('Saved meta data!')

        self.cam_1.UnregisterEventHandler(self.image_event_handler1)
        self.cam_2.UnregisterEventHandler(self.image_event_handler2)
        self.cam_1.DeInit()
        self.cam_2.DeInit()
        del self.cam_1
        del self.cam_2

    def update(self, *args):
        if self.state == 'started':
            self.num_images1.text = f"{self.image_event_handler1.cam_stem}: {self.image_event_handler1._image_count}"
            self.num_images2.text = f"{self.image_event_handler2.cam_stem}: {self.image_event_handler2._image_count}"

            if self.image_event_handler1.latest_image:
                im = self.image_event_handler1.latest_image
                texture, width, height = self.get_texture(im)
                self.image1.texture = texture
                self.image1.size = (width, height)
                self.debug_label.text = f"{width} x {height} | count: {self.image_event_handler1._image_count}"

            if self.image_event_handler2.latest_image:
                im = self.image_event_handler2.latest_image
                texture, width, height = self.get_texture(im)
                self.image2.texture = texture
                self.image2.size = (width, height)

    def get_texture(self, im):
        width = im.GetWidth()
        height = im.GetHeight()
        im_data = im.GetData()
        rgb = self.get_RGB(width, height, im_data)
        rgb_array = np.ndarray(shape=[width, height, 3], dtype=np.uint8)
        rgb_array[:, :, :] = rgb
        data = rgb_array.tobytes()
        texture = Texture.create(size=(width, height), colorfmt='rgb')
        texture.blit_buffer(data, colorfmt='rgb', bufferfmt='ubyte')
        print(f"[get_texture] Image received: {width}x{height}, dtype={rgb_array.dtype}, min={rgb_array.min()}, max={rgb_array.max()}")
        return texture, width, height

    def get_RGB(self, width, height, im_data):
        N = width * height
        fN = im_data.shape[0]
        u = np.zeros((N, ))
        v = np.zeros((N, ))
        ix4 = np.arange(0, N, 4)
        uix = np.arange(2, fN, 6)
        vix = np.arange(5, fN, 6)
        u[ix4:ix4.shape[0]*4:4] = im_data[uix]
        u[ix4+1] = im_data[uix]
        u[ix4+2] = im_data[uix]
        u[ix4+3] = im_data[uix]
        v[ix4] = im_data[vix]
        v[ix4+1] = im_data[vix]
        v[ix4+2] = im_data[vix]
        v[ix4+3] = im_data[vix]
        yix = np.unique(np.hstack((np.arange(0, fN, 6), np.arange(1, fN, 6),
                                   np.arange(3, fN, 6), np.arange(4, fN, 6))))
        assert len(yix) == N
        y = im_data[yix]
        yuv = np.dstack((np.reshape(y, [width, height]),
                         np.reshape(u, [width, height]),
                         np.reshape(v, [width, height]),))
        return self.YUV2RGB(yuv)

    def YUV2RGB(self, yuv):
        m = np.array([[1.0, 1.0, 1.0],
                      [-0.00000715478, -0.3441331, 1.7720026],
                      [1.4019976, -0.714138, 0.00001543]])
        rgb = np.dot(yuv, m)
        rgb[:, :, 0] -= 179.45477266423404
        rgb[:, :, 1] += 135.45870971679688
        rgb[:, :, 2] -= 226.8183044444304
        return rgb

    def build(self):
        root = BoxLayout(size_hint=(1, 1), orientation='horizontal')
        self.folder_string = 'test'
        self.num_saved_images1 = 'test'
        self.num_saved_images2 = 'test2'
        info = BoxLayout(size_hint=(.25, 1), height=200, orientation='vertical')
        self.folder_label = Label(text=self.folder_string)
        info.add_widget(self.folder_label)
        self.num_images1 = Label(text="Cam 1: 0")
        info.add_widget(self.num_images1)
        self.num_images2 = Label(text="Cam 2: 0")
        info.add_widget(self.num_images2)
        self.debug_label = Label(text="Waiting for image...")
        info.add_widget(self.debug_label)
        layout = BoxLayout(size_hint=(.25, 1), height=200, orientation='vertical')
        button = Button(text='set folder')
        button.bind(on_press=self.load)
        layout.add_widget(button)
        button = Button(text='start recording')
        button.bind(on_press=self.start)
        layout.add_widget(button)
        button = Button(text='stop recording')
        button.bind(on_press=self.stop)
        layout.add_widget(button)
        root.add_widget(layout)
        root.add_widget(info)
        layout2 = BoxLayout(size_hint=(.5, 1), height=200, orientation='vertical')
        self.image1 = Image()
        layout2.add_widget(self.image1)
        self.image2 = Image()
        layout2.add_widget(self.image2)
        root.add_widget(layout2)
        self.state = ''
        Clock.schedule_interval(self.update, 1./20.)
        return root

if __name__ == '__main__':
    CameraApp().run()
