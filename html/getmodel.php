<?php
$file = "./testfile.txt";
header('Content-Disposition: attachment; filename="file.txt"');
$out = `python3 serversideout.py`;
readfile($file);
exit();
?>
