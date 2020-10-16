<?php
    $config = include( "config.php" );
    $db = mysqli_connect( "localhost:3306", "root", "", "login" );
    if($db == FALSE){
        echo "DATABASE DEAD";
        exit();
    }
?>