<?php
$file = "./testfile.obj";
$out = shell_exec("sudo docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./main.py -s");
echo $out;
exit;
?>
