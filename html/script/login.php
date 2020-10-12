<?php
    
    echo "hi";
    include 'libraries.php';
    include 'webhooks.php';
    session_start();

    $badChars = array( '"', "'", "`", ";", "-" );
    $username = str_replace( $badChars, "", $_POST["user"] );
    $password = str_replace( $badChars, "", $_POST["pw"] );

    $_SESSION["user"] = $username;
    $_SESSION["loggedin"] = 0;

    include 'sql.php';
    $results = $db -> query( "SELECT * FROM Users" );

    while ( $row = mysqli_fetch_array( $results )  ) {
        if ( $row[0] == $username && $row[1] == $password ){
            send_login_attempt_webhook( $username, true );
            $_SESSION["loggedin"] = 1;
            echo $_SESSION["loggedin"];
            exit();
        } else {
            $_SESSION["loggedin"] = 0;
            send_login_attempt_webhook( $username, false );
        }
    }

    mysqli_close( $db );
?>