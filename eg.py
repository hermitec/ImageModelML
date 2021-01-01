from base64 import b64decode
from base64 import b64encode

s = b"test"
print(s)
s = b64encode(s)
# Using base64.b64decode() method
gfg = b64decode(s)

print(gfg)
