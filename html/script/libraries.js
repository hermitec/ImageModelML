var currentTime = 0;
var loginAttempts = 0;

function sleep( ms ) {
    return new Promise( resolve => setTimeout( resolve, ms ) );
}

var userInput = document.getElementById( "username" );
var pwInput = document.getElementById( "password" );

userInput.addEventListener( "keydown", function( event ) {
    if( event.keyCode === 13 ) {
        event.preventDefault();
        log_in();
    }
} );

pwInput.addEventListener( "keydown", function( event ) {
    if( event.keyCode === 13 ) {
        event.preventDefault();
        log_in();
    }
} );

async function log_in() {
    loginAttempts += 1
    await sleep(10)
    var time = Math.round( ( new Date() ).getTime() / 1000 );
    if ( time - currentTime < 5 && loginAttempts >= 3 ){
        var timeToWait = 5 - ( time-currentTime );
        document.getElementById( "errorText" ).innerHTML = "Please wait " + timeToWait + " seconds before trying to log in again.";
        return null;
    }
    currentTime = Math.round( ( new Date() ).getTime() / 1000 );
    var user = userInput.value;
    var pw = pwInput.value;
    var success = 0;

    const loginHTTPReq = new XMLHttpRequest(),
        method = "POST",
        url = "/main.php",
        params = "user=" + user + "&pw=" + pw;

    loginHTTPReq.open( method, url, true );
    loginHTTPReq.setRequestHeader( "Content-type", "application/x-www-form-urlencoded" );
    loginHTTPReq.onreadystatechange = function() {
        console.log( this.responseText );
        if( this.readyState == 4 && this.status == 200 ) {
            if( this.responseText == 1 ) window.location.href = "../mainsite.html";
            else{ document.getElementById( "errorText" ).innerHTML = "Incorrect credentials." }
        }
    }
    loginHTTPReq.send( params );
}
