<?php

$out = `docker run --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./main.py -s 2>&1`;
echo $out;
exit;
?>
