from asyncio.windows_events import NULL
from pyueye import ueye
import numpy as np
import cv2
import sys
import threading
import os

class Camera ():

    def __init__(self):
        #Variables
        self.hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.m_nColorMode = ueye.INT()		 # Y8/RGB16/RGB24/REG32
        self.nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
        self.width = 0
        self.height = 0

        #roep setupConnection op
        self.nRet = self.setupConnection()

    def setupConnection(self):
        
        # Starts the driver and establishes the connection to the camera
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")

        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        nRet = ueye.is_ResetToDefault( self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")

        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)

        # Set the right color mode
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("else")

        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        width = self.rectAOI.s32Width
        self.width = width
        height = self.rectAOI.s32Height
        self.height = height

        # Prints out some information about the camera and the sensor
        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Maximum image width:\t", width)
        print("Maximum image height:\t", height)
        print()

        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, width, height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)



        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, width, height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("Press q to leave the programm")

        return nRet

    def testCamera(self):
        # Continuous image display
        while(self.nRet == ueye.IS_SUCCESS):

            # In order to display the image in an OpenCV window we need to...
            # ...extract the data of our image memory
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)

            # bytes_per_pixel = int(nBitsPerPixel / 8)

            # ...reshape it in an numpy array...
            frame = np.reshape(array,(self.height.value, self.width.value, self.bytes_per_pixel))

            # ...resize the image by a half
            frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
            
        #---------------------------------------------------------------------------------------------------------------------------------------
            #Include image data processing here

        #---------------------------------------------------------------------------------------------------------------------------------------

            #...and finally display it
            cv2.imshow("SimpleLive_Python_uEye_OpenCV", frame)
            
            # Press q if you want to end the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #---------------------------------------------------------------------------------------------------------------------------------------
        
        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)

        # Destroys the OpenCv windows
        cv2.destroyAllWindows()

    #@return : gives back the pichture frame in Mat type
    def takePicture(self):
        if self.nRet == ueye.IS_SUCCESS :
            try:
                array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
                frame = np.reshape(array,(self.height.value, self.width.value, self.bytes_per_pixel))
                frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                return frame
            except:
                print('fout bij uitlewen van foto')

        else:
            print('fout in detectie van camera')
        return NULL
    
    def previeuw(self):
        while True:
            if self.nRet == ueye.IS_SUCCESS :
                try:
                    array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
                    frame = np.reshape(array,(self.height.value, self.width.value, self.bytes_per_pixel))
                    frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                    cv2.imshow("camera",frame)
                    cv2.waitKey(1)
                except:
                    print('fout bij uitlewen van foto')

            else:
                print('fout in detectie van camera')

c = Camera()
#c.testCamera()
thp = threading.Thread(target=c.previeuw)
thp.start()
x=1
for i in os.listdir('python/Camera/tmp_pict'):
    x+=1
count = x
while True:
    input('take picture')
    frame = c.takePicture()
    cv2.imshow('image',frame)
    cv2.waitKey(0)
    cv2.imwrite('python/Camera/tmp_pict/'+str(count)+'.jpg',frame)
    count += 1
