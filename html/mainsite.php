<?php
session_start();
if($_SESSION["loggedin"] != 1){header("http://18.222.183.14/");exit();}
?><html>
    <h1>YOU DID IT</h1>
    you cracked the code and you're in!!! 
</html>