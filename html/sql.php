<?php
    $config = include( "config.php" );
    $db = mysqli_connect("localhost", "newuser", "password", "login" );
    if (mysqli_connect_errno())
    {
    echo "Failed to connect to MySQL: " . mysqli_connect_error();
    }
?>
