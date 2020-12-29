<?php
session_start();
if($_SESSION["loggedin"] != 1){header("http://3.139.70.139");exit();}
?>
<!doctype html>
<html lang="en">
  <head>

    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <link rel="stylesheet" href="/stylesheet.css">

    <title>Image to Model</title>
  </head>
  <body>
    <form method="post">
        <div class="vertical-center">
        <div class="card mx-auto" style="width: 18rem;">
                <div class="card-body">
                    <div class="form-group">
                        <label for="username">X-1</label>
                        <input type="file" class="form-control" id="x1"name="x1">
                    </div>
                    <div class="form-group">
                        <label for="password">X-2</label>
                        <input type="file" class="form-control" id="x2" name="x2">
                    </div>
                    <p id="errorText" style="color:red; text-align: center;"></p>
                    <div class="form-group">
                        <label for="y1">Y-1</label>
                        <input type="file" class="form-control" id="y1">
                    </div>
                    <div class="form-group">
                        <label for="y2">Y-2</label>
                        <input type="file" class="form-control" id="y2">
                    </div>
                    <div class="form-group">
                        <label for="y1">Z-1</label>
                        <input type="file" class="form-control" id="y1">
                    </div>
                    <div class="form-group">
                        <label for="y2">Z-2</label>
                        <input type="file" class="form-control" id="y2">
                    </div>
                    <a href="#" class="btn btn-primary" style="width:100%;" onclick="model_compute()">Submit</a>
                    <p id="errorText" style="color:red; text-align: center;"></p>
                </div>
            </div>
        </div>
    </form>
    <iframe id="invis_iframe" style="display:none;"></iframe>
    <script>document.getElementById("x1").onchange = function(event) {
   var fileList = inputElement.files;
   console.log(fileList);  
}</script>
    <script src="./script/libraries.js"></script>
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
  </body >

</html>
