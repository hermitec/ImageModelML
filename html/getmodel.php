<?php
$out = `sudo docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./main.py -s`;
exit;
?>
