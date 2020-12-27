<?php
$file = "./testfile.obj";
$out = `sudo docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./main.py -s`;

$finfo = finfo_open(FILEINFO_MIME_TYPE);

return $finfo;
?>
