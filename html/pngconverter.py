import png, os, sys
f = open("test","r").readlines()
print(f)
rgb = []
currentPixel = []
ticker = 0 # my personal favourite bad practice
for i,x in enumerate(f):
    if ticker == 0:
        currentPixel = []
    currentPixel.append(x)
    if ticker == 2:
        rgb.append(currentPixel)

image = png.from_array(rgb)
image.save("testout.png")
