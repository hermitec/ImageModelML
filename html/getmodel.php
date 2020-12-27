<?php
$file = "./testfile.obj";
putenv('PATH=/usr/local/bin');
$out = shell_exec("sudo docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./main.py -s");
echo "hi";
exit;
?>
