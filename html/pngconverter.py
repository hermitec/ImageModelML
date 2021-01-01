import base64
from PIL import Image
from io import BytesIO

f = open("test","r")
data = f.read()
f.close()
data = bytes(data, "utf-8")
im = Image.open(BytesIO(base64.b64decode(data)))
im.save('final.png', 'PNG')
