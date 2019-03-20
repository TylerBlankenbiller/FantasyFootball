<?php require_once("ratH.html"); ?>
<tr class="pickRow">

	<td colspan = 12>
		<div class="titleCard">
			Week 12 Projections
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
					  <option value="actualSeasonLeaderRBPPR.php">RB</option>
					  <option value="actualSeasonLeaderWRPPR.php">WR</option>
					  <option value="actualSeasonLeaderTEPPR.php">TE</option>
					  <option value="actualSeasonLeaderRBPPR.php" selected = "selected">FLEX</option>
					  <option value="actualSeasonLeaderRBPPR.php">D/ST</option>
					  <option value="actualSeasonLeaderRBPPR.php">K</option>
					</select>
				</li>
			</ul>
		</div>
		<div class = "otherSelect">
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
			</div>
			<ul>
				<li class = "selectors">Week</li>
				<li>
					<select>
					  <option value="1">1</option>
					  <option value="2">2</option>
					  <option value="3">3</option>
					  <option value="1">4</option>
					  <option value="2">5</option>
					  <option value="3">6</option>
					  <option value="1">7</option>
					  <option value="2">8</option>
					  <option value="3">9</option>
					  <option value="1">10</option>
					  <option value="2">11</option>
					  <option value="3" selected = "selected">12</option>
					  <option value="1">13</option>
					  <option value="2">14</option>
					  <option value="3">15</option>
					  <option value="2">16</option>
					  <option value="3">17</option>
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