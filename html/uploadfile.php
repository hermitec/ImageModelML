<?php
session_start();  //$_POST["file"] is file to upload
$up = fopen($_POST["filename"], "w");
fwrite($up, $_POST["file"]);
fclose($up);

// I DONT KNOW WHY BUT THIS IS THE ONLY WAY I CAN GET THIS TO WORK
// IM SO SORRY
$out = `python3 pngconverter.py x1`;
echo $out;
$out = `python3 pngconverter.py x2`;
$out = `python3 pngconverter.py y1`;
$out = `python3 pngconverter.py y2`;
$out = `python3 pngconverter.py z1`;
$out = `python3 pngconverter.py z2`;
exit();
?>
