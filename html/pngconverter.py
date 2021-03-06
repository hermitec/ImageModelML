import base64
from PIL import Image
import io,sys

f = open(sys.argv[1],"r")
data = f.read()
f.close()
data = data[22:].replace(" ","+")
data = bytes(data, "utf-8")
pad = len(data)%4
data += b"="*pad
im = Image.open(io.BytesIO(base64.b64decode(data)))
w,h = im.size
if w != h:
    print("ERROR_NOTSQUARE")
im.save('./user_input/{}.png'.format(sys.argv[1]), 'PNG')
