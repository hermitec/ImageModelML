<?php

$out = `docker run -it --rm -v /var/www/html/tmp -w /tmp tensorflow/tensorflow python ./main.py -s 2>&1`;
echo $out;
exit;
?>
