<?php require_once("ratH.html"); ?>
<tr class="pickRow">
	<td colspan = 20>
        </html>
        <?php
            if(isset($_POST['submit']))
            {
                $year = $_POST['year'];
            }
            else
            {
                $year = 2018;
            }
			echo "<div class='titleCard'>Draft Projections</div>";
        ?>
        <html>
	</td>

</tr>
<tr class="pickRow">

	<td colspan = 20>
		<div class="selectOptions">
			<ul>
				<li class = "selectors">Position</li>
				<li>
                    <form action="#" method="post">
					<select name="pos" id="pos">
					  <option value="QB">QB</option>
					  <option value="RB">RB</option>
					  <option value="WR">WR</option>
					  <option value="TE">TE</option>
					  <option value="K">K</option>
					</select>
				</li>
			</ul>
		</div>
        <div class="dropDown">
			<ul>
				<li class = "selectors">Scoring Type</li>
				<li>
					<select name="points" id="points">
					  <option value="Standard">Standard</option>
					  <option value="HalfPPR">Half PPR</option>
					  <option value="PPR">PPR</option>
					</select>
				</li>
			</ul>
        </div>
        <div class="dropDown">
            <ul>
				<li class = "selectors">Year</li>
				<li>
					<select name="year" id="year">
					  <option value=2018>2018</option>
					  <option value=2017>2017</option>
					  <option value=2016>2016</option>
					  <option value=2015>2015</option>
					  <option value=2014>2014</option>
					  <option value=2013>2013</option>
					  <option value=2012>2012</option>
					  <option value=2011>2011</option>
					  <option value=2010>2010</option>
					  <option value=2009>2009</option>
					</select>
				</li>
			</ul>
        </div>
        <div class="dropDown">
            <input type="submit" name="submit" value="Update" />
            </form>
        </div>
	</td>
</tr>
    <tr>
		</html>
        <?php
            function selectOptions()
            {
                if(isset($_POST['submit']))
                {
                    $values['pos'] = $_POST['pos'];
                    $values['point'] =  $_POST['points'];
                    $values['year'] = $_POST['year'];
                }
                else
                {
                    $values['pos'] = 'QB';
                    $values['point'] =  'Standard';
                    $values['year'] = 2018;
                }
                return($values);
            }
            $values = selectOptions();
            $pos = $values['pos'];
            $point = $values['point'];
            $year = $values['year'];
            function getStats($pos, $point, $year)
            {
                $row = 1;
                $wantedColumns = array(4,6,8,9,10,12,14,23,24,25,27);
                $count = 1;
                $array = array();
                if (($handle = fopen("DataBase/allActualSeason.csv", "r")) !== FALSE) {
                    while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
                        $num = count($data);
                        $row++;
                        if($data[4] == $pos and $data[1] == $year){
                            $name = $data[2];
                            $team = $data[6];
                            if($pos=='QB')
                            {
                                $ATT = round($data[7]);//ATT
                                $COMP = round($data[8]);//Comp
                                $FUM = round($data[9]);//FUM
                                $INT = round($data[11]);//INT
                                $LST = round($data[13]);//LST
                                $RTD = round($data[21]);//RTD
                                $RYDS = round($data[22]);//RUSHYDS
                                $SACK = round($data[23]);//Sack
                                $TD = round($data[24]);//TD
                                $YDS = round($data[26]);//Throw Yards
                                $REC = round($data[18]);//REC
                                $RECTD = round($data[19]);//RECTD
                                $PATATT = round($data[14]);//PAT Att
                                $PAT_Made = round($data[15]);//PAT Made
                                $RATT = round($data[17]);//RATT
                                $RECYDS = round($data[20]);//RECYDS
                                $TGTS = round($data[25]);//TGTS
                                $FGATT = round($data[27]);//FG Att
                                $Long = round($data[28]);//Long Field Goals
                                $Med = round($data[30]);//Med Field Goals
                                $Short = round($data[32]);//Short Field Goals
                                $POINTS = ($YDS/25)+($RYDS/10)+($TD*4)+($RTD*6)-($LST*2)-($FUM*2)+($RECTD*6);
                                if($point == 'HalfPPR')
                                {
                                    $POINTS = $POINTS +(0.5 * $REC);
                                }
                                else if($point == 'PPR')
                                {
                                    $POINTS = $POINTS +$REC;
                                }
                                $array[] = array('points'=>$POINTS, 'name'=>$name, 'team'=>$team, 'att'=>$ATT, 'comp'=>$COMP, 
                                                    'fum'=>$FUM, 'int'=>$INT, 'lst'=>$LST, 'rtd'=>$RTD, 'ryds'=>$RYDS,
                                                    'sack'=>$SACK, 'td'=>$TD, 'yds'=>$YDS);
                            }
                            else if($pos=='RB' or $pos=='WR' or $pos=='TE' or $pos=='FLEX')
                            {
                                $ATT = round($data[7]);//ATT
                                $COMP = round($data[8]);//Comp
                                $FUM = round($data[9]);//FUM
                                $INT = round($data[11]);//INT
                                $LST = round($data[13]);//LST
                                $RTD = round($data[21]);//RTD
                                $RYDS = round($data[22]);//RUSHYDS
                                $SACK = round($data[23]);//Sack
                                $TD = round($data[24]);//TD
                                $YDS = round($data[26]);//Throw Yards
                                $REC = round($data[18]);//REC
                                $RECTD = round($data[19]);//RECTD
                                $PATATT = round($data[14]);//PAT Att
                                $PAT_Made = round($data[15]);//PAT Made
                                $RATT = round($data[17]);//RATT
                                $RECYDS = round($data[20]);//RECYDS
                                $TGTS = round($data[25]);//TGTS
                                $FGATT = round($data[27]);//FG Att
                                $Long = round($data[28]);//Long Field Goals
                                $Med = round($data[30]);//Med Field Goals
                                $Short = round($data[32]);//Short Field Goals
                                $POINTS = ($YDS/25)+($RYDS/10)+($TD*4)+($RTD*6)-($LST*2)-($FUM*2)+($RECTD*6);
                                if($point == 'HalfPPR')
                                    $POINTS += (0.5 * $REC);
                                else if($point == 'PPR')
                                    $POINTS += $REC;
                                $array[] = array('points'=>$POINTS, 'name'=>$name, 'team'=>$team, 'REC'=>$REC,
                                                    'TGTS'=>$TGTS, 'RECYDS'=>$RECYDS, 'RTD'=>$RTD,
                                                    'RATT'=>$RATT, 'RYDS'=>$RYDS, 'RECTD'=>$RECTD, 'fum'=>$FUM, 'lst'=>$LST,
                                                    'POINTS'=>$POINTS);
                            }
                            else if($pos=='K')
                            {
                                $FGATT = round($data[27]);//FG Att
                                $Long = round($data[28]);//Long Field Goals
                                $Med = round($data[30]);//Med Field Goals
                                $Short = round($data[32]);//Short Field Goals
                                $PATATT = round($data[14]);//PAT Att
                                $PAT_Made = round($data[15]);//PAT Made
                                $POINTS = ($Long*5)+($Med*4)+($Short*3)+$PAT_Made-($PATATT-$PAT_Made)
                                        -($PATATT-($Long+$Med+$Short));
                                $array[] = array('points'=>$POINTS, 'name'=>$name, 'team'=>$team, 'PATATT'=>$PATATT,
                                                    'PAT_Made'=>$PAT_Made, 'FGATT'=>$FGATT, 'Long'=>$Long,
                                                    'Med'=>$Med, 'Short'=>$Short);
                            }
                            $count++;
                        }
                    }
                    fclose($handle);
                    return($array);
                }
            }
                $array = getStats($pos, $point, $year);
                function retStats($array, $pos, $point, $year)
                {
                    array_multisort($array, SORT_DESC);
                    $count = 1;
                    if($pos=='QB')
                    {
                        echo '<tr class="pickRow">';
                            echo '<td colspan=2>';
                            echo '<td class = "selectors">RANK</td>';
                            echo '<td class = "selectors">PLAYER</td>';
                            echo '<td class = "selectors">TEAM</td>';
                            echo '<td class = "selectorsGreen">COMP</td>';
                            echo '<td class = "selectors">ATT</td>';
                            echo '<td class = "selectors">PCT</td>';
                            echo '<td class = "selectorsGreen">YDS</td>';
                            echo '<td class = "selectors">YDS/ATT</td>';
                            echo '<td class = "selectorsGreen">TD</td>';
                            echo '<td class = "selectorsRed">INT</td>';
                            echo '<td class = "selectors">SACK</td>';
                            echo '<td class = "selectorsGreen">RYDS</td>';
                            echo '<td class = "selectorsGreen">RTD</td>';
                            echo '<td class = "selectors">FUM</td>';
                            echo '<td class = "selectorsRed">LST</td>';
                            echo '<td class = "selectorsYellow">Points</td>';
                            echo '<td colspan=2>';
                        echo '</tr>';
                        foreach ( $array as $var ) {
                            echo "<td colspan=2>";
                            echo "<td class = 'select'>" . $count . "</td>";
                            echo "<td class = 'select'>" . $var['name'] . "</td>";
                            echo "<td class = 'select'>" . $var['team'] . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['comp'] . "</td>";
                            echo "<td class = 'select'>" . $var['att'] . "</td>";
                            if($var['att'] !=0)
                                echo "<td class = 'select'>" . round($var['comp']/$var['att'], 2, PHP_ROUND_HALF_EVEN) . "</td>";
                            else
                                echo "<td class = 'select'>0</td>";
                            echo "<td class = 'selectGreen'>" . $var['yds'] . "</td>";
                            if($var['att'] != 0)
                                echo "<td class = 'select'>" . round($var['yds']/$var['att'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            else
                                echo "<td class = 'select'>0</td>";
                            echo "<td class = 'selectGreen'>" . $var['td'] . "</td>";
                            echo "<td class = 'selectRed'>" . $var['int'] . "</td>";
                            echo "<td class = 'select'>" . $var['sack'] . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['ryds'] . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['rtd'] . "</td>";
                            echo "<td class = 'select'>" . $var['fum'] . "</td>";
                            echo "<td class = 'selectRed'>" . $var['lst'] . "</td>";
                            echo "<td class = 'selectYellow'>" . round($var['points'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            echo "<td colspan=2>";
                            echo "</tr><tr>";
                            $count++;
                        }
                    }
                    else if($pos=='TE' or $pos=='WR' or $pos=='RB' or $pos=='FLEX')
                    {
                       echo '<tr class="pickRow">';
                            echo '<td colspan=2>';
                            echo '<td class = "selectors">RANK</td>';
                            echo '<td class = "selectors">PLAYER</td>';
                            echo '<td class = "selectors">TEAM</td>';
                            echo '<td class = "selectorsGreen">REC</td>';
                            echo '<td class = "selectors">TGTS</td>';
                            echo '<td class = "selectors">Catch%</td>';
                            echo '<td class = "selectorsGreen">REC YDS</td>';
                            echo '<td class = "selectors">REC YDS/REC</td>';
                            echo '<td class = "selectorsGreen">REC TD</td>';
                            echo '<td class = "selectors">R-ATT</td>';
                            echo '<td class = "selectorsGreen">R-YDS</td>';
                            echo '<td class = "selectors">R YDS/R ATT</td>';
                            echo '<td class = "selectorsGreen">R TD</td>';
                            echo '<td class = "selectors">FUM</td>';
                            echo '<td class = "selectorsRed">LST</td>';
                            echo '<td class = "selectorsYellow">Points</td>';
                            echo '<td colspan=2>';
                        echo '</tr>';                        
                        foreach ( $array as $var ) {
                            echo "<td colspan=2>";
                            echo "<td class = 'select'>" . $count . "</td>";
                            echo "<td class = 'select'>" . $var['name'] . "</td>";
                            echo "<td class = 'select'>" . $var['team'] . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['REC'] . "</td>";
                            echo "<td class = 'select'>" . $var['TGTS'] . "</td>";
                            if($var['TGTS'] != 0)
                                echo "<td class = 'select'>" . round($var['REC']/$var['TGTS'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            else
                                echo "<td class = 'select'>0</td>";
                            echo "<td class = 'selectGreen'>" . $var['RECYDS'] . "</td>";
                            if($var['REC'] != 0)
                                echo "<td class = 'select'>" . round($var['RECYDS']/$var['REC'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            else
                                echo "<td class = 'select'>0</td>";
                            echo "<td class = 'selectGreen'>" . $var['RECTD'] . "</td>";
                            echo "<td class = 'select'>" . $var['RATT'] . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['RYDS'] . "</td>";
                            if($var['RATT'] != 0)
                                echo "<td class = 'select'>" . round($var['RYDS']/$var['RATT'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            else
                                echo "<td class = 'select'>0</td>";
                            echo "<td class = 'selectGreen'>" . $var['RTD'] . "</td>";
                            echo "<td class = 'select'>" . $var['fum'] . "</td>";
                            echo "<td class = 'selectRed'>" . $var['lst'] . "</td>";
                            echo "<td class = 'selectYellow'>" . round($var['points'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            echo "<td colspan=2>";
                            echo "</tr><tr>";
                            $count++;
                        }
                    }
                    else if($pos=='K')
                    {
                       echo '<tr class="pickRow">';
                            echo '<td colspan=3>';
                            echo '<td class = "selectors">RANK</td>';
                            echo '<td class = "selectors">PLAYER</td>';
                            echo '<td class = "selectors">TEAM</td>';
                            echo '<td class = "selectorsGreen">PAT</td>';
                            echo '<td class = "selectors">PAT ATT</td>';
                            echo '<td class = "selectors">PAT%</td>';
                            echo '<td class = "selectorsRed">PAT MISSED</td>';
                            echo '<td class = "selectorsGreen">LONG</td>';
                            echo '<td class = "selectorsGreen">MED</td>';
                            echo '<td class = "selectorsGreen">SHORT</td>';
                            echo '<td class = "selectors">FG ATT</td>';
                            echo '<td class = "selectors">FG%</td>';
                            echo '<td class = "selectorsRed">FG Missed</td>';
                            echo '<td class = "selectorsYellow">Points</td>';
                            echo '<td colspan=3>';
                        echo '</tr>';                        
                        foreach ( $array as $var ) {
                            echo "<td colspan=3>";
                            echo "<td class = 'select'>" . $count . "</td>";
                            echo "<td class = 'select'>" . $var['name'] . "</td>";
                            echo "<td class = 'select'>" . $var['team'] . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['PAT_Made'] . "</td>";
                            echo "<td class = 'select'>" . $var['PATATT'] . "</td>";
                            if($var['PATATT'] != 0)
                                echo "<td class = 'select'>" . round($var['PAT_Made']/$var['PATATT'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            else
                                echo "<td class = 'select'>0</td>";
                            echo "<td class = 'selectRed'>" . ($var['PATATT']-$var['PAT_Made']) . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['Long'] . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['Med'] . "</td>";
                            echo "<td class = 'selectGreen'>" . $var['Short'] . "</td>";
                            echo "<td class = 'select'>" . $var['FGATT'] . "</td>";
                            if($var['FGATT'] != 0)
                                echo "<td class = 'select'>" . round(($var['Long']+$var['Med']+$var['Short'])/$var['FGATT'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            else
                                echo "<td class = 'select'>0</td>";
                            echo "<td class = 'selectRed'>" . (-$var['Long']-$var['Med']-$var['Short']+$var['FGATT']) . "</td>";
                            echo "<td class = 'selectYellow'>" . round($var['points'], 1, PHP_ROUND_HALF_EVEN) . "</td>";
                            echo "<td colspan=3>";
                            echo "</tr><tr>";
                            $count++;
                        }
                    }
                }
            retStats($array, $pos, $point, $year);
        ?>
        <html>
</tr>
<?php require_once("footerT.html"); ?>