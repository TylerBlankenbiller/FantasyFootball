<?php require_once("ratH.html"); ?>
<script>
	function checkReq(e)
	{
		var msg = "";
		var focus = "";
		
		document.getElementById("missingReq2").style.visibility = "hidden";
		document.getElementById("missingReq2").style.color = "red";
		
		document.getElementById("firstNameLabel2").style.color = "black";
		document.getElementById("emailLabel2").style.color = "black";
		document.getElementById("passwordLabel2").style.color = "black";
		document.getElementById("passwordConfirmLabel2").style.color = "black";
		
		if (document.getElementById("firstName").value == "" ||
			!isNaN(parseFloat(document.getElementById("firstName").value)))
		{
			document.getElementById("firstNameLabel2").style.color = "red";
			focus = "firstName";
		}
		if ((document.getElementById("email").value == "") ||
			 (document.getElementById("email").value == "@kutztown.edu")) 
		{
			document.getElementById("emailLabel2").style.color = "red";
			if (focus == "") 
				focus = "email";
		}
		if ((document.getElementById("password").value == "") ||
			(document.getElementById("password").value.length < 8))
		{
			document.getElementById("passwordLabel2").style.color = "red";
			if (focus == "") 
				focus = "password";
		}
		if ((document.getElementById("passwordConfirm").value == "") ||
			(document.getElementById("passwordConfirm").value != document.getElementById("password").value))
		{
			document.getElementById("passwordConfirmLabel2").style.color = "red";
			if (focus == "") 
				focus = "passwordConfirm";
		}
		if (focus != "") 
		{
			document.getElementById("missingReq2").style.visibility = "visible";
			document.getElementById(focus).focus();
			window.scrollTo(0,0);
			e.preventDefault();
			return false;
		}
		window.location.href = "log.php"
	}
	
	function signIn(e)
	{
		document.getElementById("missingReq").style.visibility = "hidden";
		document.getElementById("missingReq").style.color = "red";
		
		document.getElementById("firstNameLabel").style.color = "black";
		document.getElementById("passwordLabel").style.color = "black";
		
		var focus = "";
		
		if (document.getElementById("Name").value == "")
		{
			document.getElementById("firstNameLabel").style.color = "red";
			focus = "Name";
		}
		if (document.getElementById("pass").value == "")
		{
			document.getElementById("passwordLabel").style.color = "red";
			if (focus == "") 
				focus = "pass";
		}
		if (focus != "") 
		{
			document.getElementById("missingReq").style.visibility = "visible";
			document.getElementById(focus).focus();
			window.scrollTo(0,0);
			e.preventDefault();
			return false;
		}
		window.location.href = "log.php"
	}
	
	function clearForm()  
	{
		document.getElementById("missingReq2").style.visibility = "hidden";
		document.getElementById("firstNameLabel2").style.color = "black";
		document.getElementById("emailLabel2").style.color = "black";
		document.getElementById("passwordLabel2").style.color = "black";
		document.getElementById("passwordConfirmLabel2").style.color = "black";
		document.getElementById("firstName").value ="";
		document.getElementById("email").value ="";
		document.getElementById("password").value ="";
		document.getElementById("passwordConfirm").value ="";
	}
	function clearF()  
	{
		document.getElementById("missingReq").style.visibility = "hidden";
		document.getElementById("firstNameLabel").style.color = "black";
		document.getElementById("passwordLabel").style.color = "black";
		document.getElementById("Name").value ="";
		document.getElementById("pass").value ="";
	}
	
	
	//window.addEventListener( "submit", checkReq, false);
	//window.addEventListener( "signIn", signIn, false);
    //window.addEventListener( "reset", clearForm, false);
</script>

<tr class="leaderRow">
	<td colspan=6>
		<div class="signIn">Sign In</div>
		<form method="post" action="log.php" id="signin">
			<input type="hidden" id="focusEl"/>
			<div id ="missingReq" class = "reqHid" style="visibility:hidden">Incorrect Username/Password</div>
			<div class = "reqTop" id = "firstNameLabel">UserName or Email:</div>
				<div class="createField"><input type="text" name="Name" id = "Name" size="30" maxlength="30" /></div>
			<div class = "req" id= "passwordLabel">Password:</div>
				<div class="createField"><input type="password" name="pass" id="pass" size="30" maxlength="50" /></div>
			<br/>
			
			<div class="createField">
			<!--<input type="submit" name="signIn" value="Sign In"/>-->
			<button type="button" onclick="signIn()">SignIn</button>
			<button type="button" onclick="clearF()">Clear</button>
			</div>
		</form>
		<br/><br/>
	</td>
	<td colspan=6>
		<div class="createAcc">Create Account</div>
		<form method="post" action="log.php" id="register">
			<input type="hidden" id="focusEl"/>
			<div id ="missingReq2" class = "reqHid" style="visibility:hidden">Fields in red are Required/Incorrect Input</div>
			<div class = "reqTop" id = "firstNameLabel2">UserName:</div>
				<div class="createField"><input type="text" name="firstName" id = "firstName" size="30" maxlength="30" /></div>
			<div class = "req" id= "emailLabel2">E-mail Address:</div>
				<div class="createField"><input type="text" name="email" id="email" size="30" maxlength="50" /></div>
			<div class = "req" id= "passwordLabel2">Password:</div>
				<div class="createField"><input type="password" name="password" id="password" size="30" maxlength="50" /></div>
			<div class = "req" id= "passwordConfirmLabel2">Confirm Password:</div>
				<div class="createField"><input type="password" name="passwordConfirm" id="passwordConfirm" size="30" maxlength="50" /></div><br/>
			
			<div class="createField">
				<!--<input type="submit" name="submit" value="Create Account"/>-->
				<button type="button" onclick="checkReq()">Create Account</button>
				<button type="button" onclick="clearForm()">Clear</button>
			</div>
		</form>
		<br/><br/>
	</td>
</tr>
<?php require_once("footerT.html"); ?>