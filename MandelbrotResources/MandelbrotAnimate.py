from PIL import Image, ImageTk
import numpy as np
from numba import cuda
import cv2
import time

start = time.time()

width = 1920
height = 1080

ratio = width/height
centreR = -0.8194969120201987
centreI = 0.20310840552178341

xRange = 4
yRange = xRange/ratio

iterations = 2000


@cuda.jit
def generate_pixel_data(output_array_r, output_array_g, output_array_b, xRange, yRange, centreR, centreI, iterations):
    j, i = cuda.grid(2)
    if j < output_array_r.shape[0] and i < output_array_r.shape[1]:
        zR = i/width*xRange - xRange/2 + centreR
        zI = -j/height*yRange + yRange/2 + centreI
        
        zRNew = zR
        zINew = zI
        inSet = True
        for k in range(1, iterations+1):
            zRNewt = zRNew**2 - zINew**2 + zR
            zINew = 2*zRNew*zINew + zI
            zRNew = zRNewt
            if zINew**2 + zRNew**2 > 4:
                inSet = False
                hue = k % 360
                break
            
        if inSet == True:
            output_array_r[j, i] = 0
            output_array_g[j, i] = 0
            output_array_b[j, i] = 0
        else:
            val = hue % 60
            band = hue//60
            if band == 0:
                r = 255
                g = val*255/60
                b = 0
            elif band == 1:
                r = 255 - val*255/60
                g = 255
                b = 0
            elif band == 2:
                r = 0
                g = 255
                b = val*255/60
            elif band == 3:
                r = 0
                g = 255 - val*255/60
                b = 255
            elif band == 4:
                r = val*255/60
                g = 0
                b = 255
            elif band == 5:
                r = 255
                g = 0
                b = 255 - val*255/60
                
            output_array_r[j, i] = r
            output_array_g[j, i] = g
            output_array_b[j, i] = b



def generate_frame():
    global gif
    
    array_gpu_r = cuda.device_array((height, width), dtype=np.float64)
    array_gpu_g = cuda.device_array((height, width), dtype=np.float64)
    array_gpu_b = cuda.device_array((height, width), dtype=np.float64)
    
    threadsperblock = (16, 16)
    blockspergrid = (array_gpu_r.shape[0] // threadsperblock[0], array_gpu_r.shape[1] // threadsperblock[1])
    generate_pixel_data[blockspergrid, threadsperblock](array_gpu_r, array_gpu_g, array_gpu_b, xRange, yRange, centreR, centreI, iterations)
    
    array_red = array_gpu_r.copy_to_host()
    array_green = array_gpu_g.copy_to_host()
    array_blue = array_gpu_b.copy_to_host()
    
    array_red = np.uint8(array_red)
    array_green = np.uint8(array_green)
    array_blue = np.uint8(array_blue)
        
    array_rgb = np.stack((array_red, array_green, array_blue), axis=-1)
    image = Image.fromarray(array_rgb)

    gif.append(image)

def zoom():
    global xRange, yRange
    xRange = xRange*0.96235
    yRange = xRange/ratio
    generate_frame()

gif = []    

generate_frame()

for i in range(1, 841):
    zoom()
    print(f"{round(i/8.4,2)}%")




#Create Ouput Video
output_video = 'output_video.mp4'


fourcc = cv2.VideoWriter_fourcc(*'LAGS')
video_writer = cv2.VideoWriter(output_video, fourcc, 60.0, (width, height))

for frame in gif:
    frame_np = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    video_writer.write(frame_np)

video_writer.release()

end = time.time()
print(f"took {end-start} seconds to render")
