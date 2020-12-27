<?php
$file = "./testfile.obj";
putenv('PATH=/usr/local/bin');
$debg = `sudo python3 ./serversideout.py`;
$out = `sudo docker run -it --rm -v /tmp -w /tmp tensorflow/tensorflow python ./main.py -s`;
echo "hi";
echo $debg;
exit;
?>
