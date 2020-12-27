<?php
$file = "./testfile.obj";
putenv('PATH=/usr/local/bin');
$out = `sudo docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow /usr/bin/python ./main.py -s`;
echo "hi";
exit;
?>
