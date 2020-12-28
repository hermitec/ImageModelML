<?php

$out = `docker run --rm --mount type=bind,src=/var/www/html,dst=/tmp tensorflow/tensorflow python ./tmp/main.py -s 2>&1`;
echo $out;
exit;
?>
