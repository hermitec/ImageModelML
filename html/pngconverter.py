import base64
from PIL import Image
from io import BytesIO

f = open("test","r")
data = f.read()
f.close()

def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data, altchars)
im = Image.open(BytesIO(decode_base64(data[22:])))
im.save('final.png', 'PNG')
