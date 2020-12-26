<?php
$file = "./testfile.txt";
header('Content-Disposition: attachment; filename="file.txt"');
$out = `python3 /main.py -s`;
readfile($file);
exit();
?>
