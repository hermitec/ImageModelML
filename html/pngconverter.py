import base64
from PIL import Image
import io

f = open(sys.argv[0],"r")
data = f.read()
f.close()
data = data[22:].replace(" ","+")
data = bytes(data, "utf-8")
pad = len(data)%4
data += b"="*pad
im = Image.open(io.BytesIO(base64.b64decode(data)))
im.save('{}.png'.format(sys.argv[0]), 'PNG')
