'''
Mesh test
=========

This demonstrates the use of a mesh mode to distort an image. You should see
a line of buttons across the bottom of a canvas. Pressing them displays
the mesh, a small circle of points, with different mesh.mode settings.
'''

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
    """
    This class defines the properties, parameters, and the event handler itself. Take a
    moment to notice what parts of the class are mandatory, and what have been
    added for demonstration purposes. First, any class used to define image event handlers
    must inherit from ImageEventHandler. Second, the method signature of OnImageEvent()
    must also be consistent. Everything else - including the constructor,
    destructor, properties, body of OnImageEvent(), and other functions -
    is particular to the example.
    """

    def __init__(self, cam, folder_path, cam_stem):
        """
        Constructor. Retrieves serial number of given camera and sets image counter to 0.

        :param cam: Camera instance, used to get serial number for unique image filenames.
        :type cam: CameraPtr
        :rtype: None
        """
        super(ImageEventHandler, self).__init__()

        # Initialize image counter to 0
        self._image_count = 0

        # Save this for later 
        self.folder_path = folder_path 
        self.cam_stem = cam_stem

        self.i1 = []; self.t1 = []; self.id1 = []; 
        self.T = []; self.T0 = time.time()
        self.latest_image = None
        # Release reference to camera
        # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
        # cleaned up when going out of scope.
        # The usage of del is preferred to assigning the variable to None.
        del cam

    def OnImageEvent(self, image):
        """
        This method defines an image event. In it, the image that triggered the
        event is converted and saved before incrementing the count. Please see
        Acquisition example for more in-depth comments on the acquisition
        of images.

        :param image: Image from event.
        :type image: ImagePtr
        :rtype: None
        """
        # Save max of _NUM_IMAGES Images
        # Check if image is incomplete
        if image.IsIncomplete():
            print('Image incomplete with image status %i...' % image.GetImageStatus())

        else:
            # Get chunk data and save image     
            # Acquire images
            
            cd1 = image.GetChunkData()
            self.t1.append(cd1.GetTimestamp())
            self.id1.append(image.GetFrameID())
            # Save images
            z = str(self._image_count) # string 
            zz = z.zfill(6) # 000 fills 

            # images 
            image.Save(os.path.join(self.folder_path, '%s_%s.raw'%(self.cam_stem, zz)))

            self.T.append(time.time() - self.T0)

            self._image_count += 1

            self.latest_image = image; 

    def get_image_count(self):
        """
        Getter for image count.

        :return: Number of images saved.
        :rtype: int
        """
        return self._image_count

class CameraApp(App):

    def load(self, *args): 
        
        # Create folder 
        now = datetime.now()
        folder_str = now.strftime("%Y_%m_%d_%H_%M_%S")

        # Make folder string
        self.folder_path = os.path.join("S:/", folder_str)

        self.folder_label.text = self.folder_path

        # Make OS
        os.mkdir(self.folder_path)

        self.state = 'loaded'

    def start(self, *args): 

        if self.state == 'loaded': 

            # Set camera serial numbers
            serial_1 = '22157398' # side
            serial_2 = '22157271' # top 
             
            # Get system
            # # Get camera list
            cam_list = system.GetCameras()
             
            # Get cameras by serial
            self.cam_1 = cam_list.GetBySerial(serial_1)
            self.cam_2 = cam_list.GetBySerial(serial_2)
             
            # Initialize cameras
            self.cam_1.Init()
            self.cam_2.Init()

            # Set up primary camera trigger to secondary camera trigger
            self.cam_1.LineSelector.SetValue(PySpin.LineSelector_Line2)
            self.cam_1.V3_3Enable.SetValue(True)

            # # Set up secondary camera trigger
            self.cam_2.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            self.cam_2.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
            self.cam_2.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
            self.cam_2.TriggerMode.SetValue(PySpin.TriggerMode_On)

            # Setup primary to be  
            self.cam_1.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            self.cam_1.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
            self.cam_1.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
            self.cam_1.TriggerMode.SetValue(PySpin.TriggerMode_On)

            # # Set acquisition mode to acquire a single frame, this ensures acquired images are sync'd since camera 2 and 3 are setup to be triggered
            self.cam_1.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
            self.cam_2.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
             
            ## Setup buffer mode ##
            # Retrieve Stream Parameters device nodemap
            s_node_map = self.cam_1.GetTLStreamNodeMap()

            # Set buffer count to max; 
            # buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
            # # Buffer max 
            # print("setting to max buffer: %d"%buffer_count.GetMax())
            # buffer_count.SetValue(buffer_count.GetMax())

            # # Retrieve Buffer Handling Mode Information
            # handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
            # handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
            # handling_mode.SetIntValue(handling_mode_entry.GetValue())
            # print('\n\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())


            ## Setup handler 
            self.image_event_handler1 = ImageEventHandler(self.cam_1, self.folder_path, 'cam_1')
            self.image_event_handler2 = ImageEventHandler(self.cam_2, self.folder_path, 'cam_2')

            self.cam_1.RegisterEventHandler(self.image_event_handler1)
            self.cam_2.RegisterEventHandler(self.image_event_handler2)


            # Start acquisition; note that secondary cameras have to be started first so acquisition of primary camera triggers secondary cameras.
            self.cam_2.BeginAcquisition()
            self.cam_1.BeginAcquisition()

            self.wait_for_images(self.image_event_handler1)
            self.wait_for_images(self.image_event_handler2)
            
            self.state = 'started'
        else: 
            print('State is %s, not loaded'%self.state)

    def wait_for_images(self, handler): 
        while self.state == 'started':
            pass 

    def stop(self, *args):

        self.state = 'stop'
        # end acquisition
        self.cam_1.EndAcquisition()
        self.cam_2.EndAcquisition()

        # save files
        d = dict(t1=self.image_event_handler1.t1, t2=self.image_event_handler2.t1,
            t1_backup=self.image_event_handler1.T, t2_backup=self.image_event_handler2.T,
            f1 = self.image_event_handler1.id1, f2=self.image_event_handler2.id1)

        sio.savemat(os.path.join(self.folder_path, "meta_data.mat"), d)
        print('Saved meta data! ')
        print(os.path.join(self.folder_path, "meta_data.mat"))
        
        # Cleanup cameras 
        self.cam_1.UnregisterEventHandler(self.image_event_handler1)
        self.cam_2.UnregisterEventHandler(self.image_event_handler2)

        # De init 
        self.cam_1.DeInit()
        self.cam_2.DeInit()

        # Delete object 
        del self.cam_1
        del self.cam_2

    def update(self, *args): 
        if self.state == 'started': 
            self.num_images1.text = "%s: %d"%(self.image_event_handler1.cam_stem, self.image_event_handler1._image_count)
            self.num_images2.text = "%s: %d"%(self.image_event_handler2.cam_stem, self.image_event_handler2._image_count)

            if self.image_event_handler1.latest_image is None:
                pass
            else: 
                im = self.image_event_handler1.latest_image

                ### Get the texture and size params 
                texture, width, height = self.get_texture(im) 

                self.image1.texture=texture
                self.image1.size=(width, height)

            if self.image_event_handler2.latest_image is None:
                pass
            else: 
                im = self.image_event_handler2.latest_image
                
                texture, width, height = self.get_texture(im) 
                self.image2.texture=texture
                self.image2.size=(width, height)            

    def get_texture(self, im): 
        '''
        Get the width / height / RGB matrix from the camera data
        '''

        width = im.GetWidth()
        height = im.GetHeight()
        
        ## Make a texture
        im_data = im.GetData()

        rgb = self.get_RGB(width, height, im_data)
        rgb_array = np.ndarray(shape=[width, height, 3], dtype=np.uint8)
        rgb_array[:, :, :] = rgb 

        data = rgb_array.tostring()

        texture = Texture.create(size=(width, height), colorfmt='rgb')

        texture.blit_buffer(data, colorfmt='rgb', bufferfmt='ubyte')
        return texture, width, height

    def get_RGB(self, width, height, im_data): 
        N = width*height 
        fN = im_data.shape[0]

        ## Reshape 
        u = np.zeros((N, ))
        v = np.zeros((N, ))

        ix4 = np.arange(0, N, 4)
        uix = np.arange(2, fN, 6)
        vix = np.arange(5, fN, 6)

        u[ix4] = im_data[uix]
        u[ix4+1] = im_data[uix]
        u[ix4+2] = im_data[uix]
        u[ix4+3] = im_data[uix]
        
        v[ix4] = im_data[vix]
        v[ix4+1] = im_data[vix]
        v[ix4+2] = im_data[vix]
        v[ix4+3] = im_data[vix]

        yix = np.unique(np.hstack((np.arange(0, fN, 6),
                                   np.arange(1, fN, 6),
                                   np.arange(3, fN, 6), 
                                   np.arange(4, fN, 6))))

        assert(len(yix) == N)
        
        y = im_data[yix]

        yuv = np.dstack(( np.reshape(y, [width, height]), 
                          np.reshape(u, [width, height]),
                          np.reshape(v, [width, height]),))
        
        return self.YUV2RGB(yuv)

    def YUV2RGB(self, yuv ):
        '''
        from: https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
        '''
        m = np.array([[ 1.0, 1.0, 1.0],
                     [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                     [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
        
        rgb = np.dot(yuv,m)
        rgb[:,:,0]-=179.45477266423404
        rgb[:,:,1]+=135.45870971679688
        rgb[:,:,2]-=226.8183044444304
        return rgb

    def build(self):

        root = BoxLayout(size_hint = (1, 1), orientation='horizontal')

        self.folder_string = 'test'
        self.num_saved_images1 = 'test'
        self.num_saved_images2 = 'test2'

        # Information
        info = BoxLayout(size_hint = (.25, 1), height=200,
            orientation = 'vertical')
        self.folder_label = Label(text=self.folder_string)
        info.add_widget(self.folder_label)
        
        self.num_images1 = Label(text="Cam 1: 0")
        info.add_widget(self.num_images1)
        
        self.num_images2 = Label(text="Cam 2: 0")
        info.add_widget(self.num_images2)
        

        ## Buttons
        layout = BoxLayout(size_hint=(.25, 1), height=200,
            orientation = 'vertical')

        button = Button(text='set folder')
        button.bind(on_press=self.load)
        layout.add_widget(button)

        button = Button(text='start recording')
        button.bind(on_press = self.start)
        layout.add_widget(button)

        button = Button(text='stop recording')
        button.bind(on_press = self.stop)
        layout.add_widget(button)
        root.add_widget(layout)
        root.add_widget(info)
        

        # ## Cam image
        layout2 = BoxLayout(size_hint = (.5, 1), height=200, orientation='vertical')
        self.image1 = Image()
        layout2.add_widget(self.image1)
        self.image2 = Image()
        layout2.add_widget(self.image2)
        root.add_widget(layout2)

        ## clock 
        self.state = ''
        Clock.schedule_interval(self.update, 1./20.)

        return root

if __name__ == '__main__':
    CameraApp().run()