<?php require_once("ratH.html"); ?>
<tr class="pickRow">

	<td colspan = 12>
		<div class="titleCard">
			Season Leaders
		</div>
	</td>

</tr>
<tr class="pickRow">

	<td colspan = 12>
		<div class="selectOptions">
			<ul>
				<li class = "selectors">Position</li>
				<li>
					<select onchange="location = this.options[this.selectedIndex].value;">
					  <option value="actualSeasonLeaderQBPPR.php">QB</option>
					  <option value="actualSeasonLeaderRBPPR.php" selected = "selected">RB</option>
					  <option value="actualSeasonLeaderWRPPR.php">WR</option>
					  <option value="actualSeasonLeaderTEPPR.php">TE</option>
					  <option value="actualSeasonLeaderQBPPR.php">FLEX</option>
					  <option value="actualSeasonLeaderQBPPR.php">D/ST</option>
					  <option value="actualSeasonLeaderQBPPR.php">K</option>
					</select>
				</li>
			</ul>
		</div>
			<ul>
				<li class = "selectors">Scoring Type</li>
				<li>
					<select>
					  <option value="Standard">Standard</option>
					  <option value="HalfPPR">Half PPR</option>
					  <option value="PPR" selected = "selected">PPR</option>
					</select>
				</li>
			</ul>
	</td>
</tr>
<tr class="pickRow">
	<td colspan = 12 class = "table">
		<img src="Pics/RBReal.PNG" height="225" width="1234">
	</td>
</tr>
<?php require_once("footerT.html"); ?>