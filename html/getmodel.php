<?php
if function_exists('shell_exec'){
  echo "yes";
}
else{echo "no";}
$out = `sudo docker run -it --rm -v /tmp -w /tmp tensorflow/tensorflow python ./main.py -s`;
exit;
?>
