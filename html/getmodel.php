<?php
$file = "./testfile.obj";
$out = `sudo docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./main.py -s`;
return "it should have worked? i think? uhhh";
exit;
?>
