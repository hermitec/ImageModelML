<?php
    $config = include( "config.php" );
    $db = mysqli_connect( $config["database"]["host"], $config["database"]["user"], $config["database"]["password"], $config["database"]["name"] );
?>