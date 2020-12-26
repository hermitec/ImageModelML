<?php
$file = "./testfile.obj";
header('Content-Disposition: attachment; filename="file.txt"');
$out = `python3 /main.py -s`;
readfile($file);
exit();
?>
