<?php
    $config = include( "config.php" );
    $db = mysqli_connect( "localhost", "newuser", "password", "login" );
    if($db == FALSE){
        echo "DATABASE DEAD";
        exit();
    }
?>