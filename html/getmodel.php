<?php
$file = "./testfile.txt";
header('Content-Disposition: attachment; filename="'.basename($file).'"');
$out = `python3 serversideout.py`;
readfile($file);
exit();
?>
