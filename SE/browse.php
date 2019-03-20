<?php require_once("headerTH.html"); ?>
<script>
	function clearForm(e)  
	{
		document.getElementById("search").value() = "";
	}
	function checkReq(e)
	{
		var focus = "";
		if(document.getElementById("search").value == "")
		{
			e.preventDefault();
			document.getElementById("searchLabel").style.color = "red";
			focus = "search";
			document.getElementById(focus).focus();
		}
		return true;
	}
	window.addEventListener( "submit", checkReq, false);
    window.addEventListener( "reset", clearForm, false);
</script>
	<div class="title">Search or Browse by Product Category</div>
	<tr>
		<td colspan = 2>
			<div class ="title">Product Categories</div>
		</td>
		<td colspan = 3>
			<div class ="title">Search for a Product</div>
		</td>
	</tr>
	<tr>
		<td class="title" colspan = 2>
			<a href ="dairy.php">Dairy</a><br/><br/>
			<a href ="Produce.php">Produce</a><br/><br/>
			<a href ="beverages.php">Beverages</a><br/><br/>
			<a href ="Meats.php">Meats</a><br/><br/>
			<a href ="Baking.php">Baking</a><br/><br/>
			<a href ="Packaged.php">Packaged</a><br/><br/>
		</td>
		<td colspan = 3>
			<form method="post" action="search.php">
				<div class="search" id="searchLabel">Enter a search string:<br/><br/>
					<input type="text" name="search" id = "search" size="20" maxlength="30" /><br/><br/>
					<input type="submit" name="submit" value="Search"/>
					<input type="reset" value="Clear"/>
				</div>
			</form>
<?php require_once("footerT.html"); ?>