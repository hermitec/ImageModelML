<?php
$file = "./testfile.obj";
putenv('PATH=/usr/local/bin');
$out = `sudo docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./main.py -s`;
echo "hi";
exit;
?>
