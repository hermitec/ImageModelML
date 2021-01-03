<?php
session_start();  //$_POST["file"] is file to upload
$up = fopen($_POST["filename"], "w");
fwrite($up, $_POST["file"]);
fclose($up);
$out = shell_exec("python3 pngconverter.py " + $_POST["filename"]);
?>
