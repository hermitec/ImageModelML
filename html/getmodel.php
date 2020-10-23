<?php
$file = "./testfile.txt";
header('Content-Description: File Transfer');
header('Content-Type: text/html; charset=UTF-8');
header('Content-Disposition: attachment; filename="'.basename($file).'"');
header('Expires: 0');
header('Cache-Control: must-revalidate');
header('Pragma: public');
header('Content-Length: ' . filesize($file));
$out = `python3 serversideout.py`;
readfile($file);
exit();
?>
