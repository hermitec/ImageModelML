<?php
    
    session_start();

    $badChars = array( '"', "'", "`", ";", "-" );
    $username = str_replace( $badChars, "", $_POST["user"] );
    $password = str_replace( $badChars, "", $_POST["pw"] );

    $_SESSION["user"] = $username;
    $_SESSION["loggedin"] = 0;

    include 'sql.php';
    if(!$db){echo"bruh";exit();}

    $results = $db -> query( "SELECT * FROM login" );

    while ( $row = mysqli_fetch_array( $results )  ) {
        if ( $row[0] == $username && $row[1] == $password ){
            $_SESSION["loggedin"] = 1;
            echo $_SESSION["loggedin"];
            exit();
        } else {
            $_SESSION["loggedin"] = 0;
        }
    }

    mysqli_close( $db );
?>