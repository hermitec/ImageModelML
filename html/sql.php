<?php
    $config = include( "config.php" );
    $db = mysqli_connect( $config["database"]["host"], $config["database"]["user"], $config["database"]["password"], $config["database"]["name"] );
    if (!$db) {
    echo "Error: Unable to connect to MySQL." . PHP_EOL;
    echo "Debugging errno: " . mysqli_connect_errno() . PHP_EOL;
    echo "Debugging error: " . mysqli_connect_error() . PHP_EOL;
    exit;
}
?>
