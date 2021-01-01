import base64
from PIL import Image
from io import BytesIO

f = open("test","r")
data = f.read()
f.close()
data = data[22:]
data = bytes(data, "utf-8")
pad = len(data)%4
data += b"="*pad
im = Image.open(BytesIO(base64.b64decode(data)))
im.save('final.png', 'PNG')
