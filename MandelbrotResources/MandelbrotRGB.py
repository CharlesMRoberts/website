from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from numba import cuda

window = Tk()
window.attributes("-fullscreen", True)
window.title("mandelbrot")
width = window.winfo_screenwidth()
height = window.winfo_screenheight()
canvas = Canvas(window, width=width, height=height, highlightthickness=0)

ratio = width/height

centreR = 0
centreI = 0

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



def draw_image():
    global photo
    
    array_gpu_r = cuda.device_array((height, width), dtype=np.float32)
    array_gpu_g = cuda.device_array((height, width), dtype=np.float32)
    array_gpu_b = cuda.device_array((height, width), dtype=np.float32)
    
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
    photo = ImageTk.PhotoImage(image)
    
    image.save("mandelbrot.png")
    
    canvas.create_image(0, 0, anchor=NW, image=photo)
    canvas.update()

draw_image()

#Create Buttons
def zoomInClick():
    global xRange, yRange
    xRange = xRange*0.75
    yRange = xRange/ratio
    draw_image()

def zoomOutClick():
    global xRange, yRange
    xRange = xRange/0.75
    yRange = xRange/ratio
    draw_image()

def panLeftClick():
    global centreR, xRange
    centreR = centreR - xRange*0.25
    draw_image()
    
def panRightClick():
    global centreR, xRange
    centreR = centreR + xRange*0.25
    draw_image()

def panUpClick():
    global centreI, yRange
    centreI = centreI + yRange*0.25
    draw_image()

def panDownClick():
    global centreI, yRange
    centreI = centreI - yRange*0.25
    draw_image()

def setIterations():
    global iterations
    iterations = int(iterationsInput.get(1.0, END))
    draw_image()

canvas.pack(fill=BOTH, expand=YES)

zoomIn = Button(window, text="+", command=zoomInClick)
zoomIn.config(width=1, height=1)
zoomIn.place(x=0.9*1920, y=0.55*1080)

zoomOut = Button(window, text="-", command=zoomOutClick)
zoomOut.config(width=1, height=1)
zoomOut.place(x=0.95*1920, y=0.55*1080)

panLeft = Button(window, text="←", command=panLeftClick)
panLeft.config(width=1, height=1)
panLeft.place(x=0.91*1920, y=0.45*1080)

panRight = Button(window, text="→", command=panRightClick)
panRight.config(width=1, height=1)
panRight.place(x=0.94*1920, y=0.45*1080)

panDown = Button(window, text="↓", command=panDownClick)
panDown.config(width=1, height=1)
panDown.place(x=0.925*1920, y=0.45*1080)

panUp = Button(window, text="↑", command=panUpClick)
panUp.config(width=1, height=1)
panUp.place(x=0.925*1920, y=0.425*1080)

iterationsInput = Text(window, width=5, height=1)
iterationsInput.place(x=0.925*1920, y=0.6*1080)

iterationsSet = Button(window, text="set", command=setIterations)
iterationsSet.config(width=1,height=1)
iterationsSet.place(x=0.915*1920, y=0.6*1080)

window.mainloop()
