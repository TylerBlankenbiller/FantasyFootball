<?php require_once("ratH.html"); ?>
<tr class="pickRow">
	<td colspan = 20>
        </html>
        <?php
            if(isset($_POST['submit']))
            {
                $week_val = $_POST['Week'];
            }
            else
            {
                $week_val = 1;
            }
			echo "<div class='titleCard'>Week " . $week_val . " Projections</div>";
        ?>
        <html>
	</td>

</tr>
<tr class="pickRow">

	<td colspan = 20>
		<div class="selectOptions">
			<ul>
				<li class = "selectors">Position</li>
				<li><!-- Select Options -->
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
				<li class = "selectors">Week</li>
				<li>
					<select name='Week' id="Week">
					  <option value=1>1</option>
					  <option value=2>2</option>
					  <option value=3>3</option>
					  <option value=4>4</option>
					  <option value=5>5</option>
					  <option value=6>6</option>
					  <option value=7>7</option>
					  <option value=8>8</option>
					  <option value=9>9</option>
					  <option value=10>10</option>
					  <option value=11>11</option>
					  <option value=12>12</option>
					  <option value=13>13</option>
					  <option value=14>14</option>
					  <option value=15>15</option>
					  <option value=16>16</option>
					  <option value=17>17</option>
					</select>
                    <input type="submit" name="submit" value="Update" />
                    </form>
				</li>
			</ul>
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
                    $values['week_val'] = $_POST['Week'];
                    $values['pos'] = $_POST['pos'];
                    $values['point'] =  $_POST['points'];
                    #$week_val = $_POST['Week'];
                    #$pos = $_POST['pos'];
                    #$point = $_POST['points'];
                    #$year = $_POST['year'];
                }
                else
                {
                    $values['week_val'] = 1;
                    $values['pos'] = 'QB';
                    $values['point'] =  'Standard';
                    #$week_val = 1;
                    #$pos='QB';
                    #$point = 'Standard';
                    #$year = 2018;
                }
                return($values);
            }
            $values = selectOptions();
            $week_val = $values['week_val'];
            $pos = $values['pos'];
            $point = $values['point'];
            function getStats($week_val, $pos, $point)
            {
                $row = 1;
                $wantedColumns = array(4,6,8,9,10,12,14,23,24,25,27);
                $count = 1;
                $array = array();
                if (($handle = fopen("DataBase/allActualStats.csv", "r")) !== FALSE) {
                    while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
                        $num = count($data);
                        $row++;
                        if($data[$num-1] == $pos and $data[1]==$week_val){
                            $name = $data[4];
                            $team = $data[6];
                            if($pos=='QB')
                            {
                                $ATT = round($data[8]);//ATT
                                $COMP = round($data[9]);//Comp
                                $FUM = round($data[10]);//FUM
                                $INT = round($data[12]);//INT
                                $LST = round($data[14]);//LST
                                $RTD = round($data[22]);//RTD
                                $RYDS = round($data[23]);//RECYDS
                                $SACK = round($data[24]);//Sack
                                $TD = round($data[25]);//TD
                                $YDS = round($data[27]);//Throw Yards
                                $REC = round($data[19]);//REC
                                $RECTD = round($data[20]);//RECTD
                                $PATATT = round($data[15]);//PAT Att
                                $PAT_Made = round($data[16]);//PAT Made
                                $RATT = round($data[18]);//RATT
                                $RECYDS = round($data[21]);//RECYDS
                                $TGTS = round($data[26]);//TGTS
                                $FGATT = round($data[28]);//FG Att
                                $Long = round($data[29]);//Long Field Goals
                                $Med = round($data[31]);//Med Field Goals
                                $Short = round($data[33]);//Short Field Goals
                                $POINTS = ($YDS/25)+($RYDS/10)+($TD*4)+($RTD*6)-($LST*2)-($FUM*2)+($RECTD*6);
                                if($point = 'HalfPPR')
                                    $POINTS += (0.5 * $REC);
                                else if($point = 'PPR')
                                    $POINTS += $REC;
                                $array[] = array('points'=>$POINTS, 'name'=>$name, 'team'=>$team, 'att'=>$ATT, 'comp'=>$COMP, 
                                                    'fum'=>$FUM, 'int'=>$INT, 'lst'=>$LST, 'rtd'=>$RTD, 'ryds'=>$RYDS,
                                                    'sack'=>$SACK, 'td'=>$TD, 'yds'=>$YDS);
                            }
                            else if($pos=='RB' or $pos=='WR' or $pos=='TE' or $pos=='FLEX')
                            {
                                $COMP = round($data[9]);//Comp
                                $FUM = round($data[10]);//FUM
                                $INT = round($data[12]);//INT
                                $LST = round($data[14]);//LST
                                $RTD = round($data[22]);//RTD
                                $RYDS = round($data[23]);//RECYDS
                                $TD = round($data[25]);//TD
                                $YDS = round($data[27]);//Throw Yards
                                $REC = round($data[19]);//REC
                                $RECTD = round($data[20]);//RECTD
                                $RATT = round($data[18]);//RATT
                                $RECYDS = round($data[21]);//RECYDS
                                $TGTS = round($data[26]);//TGTS
                                $POINTS = ($YDS/25)+($RYDS/10)+($TD*4)+($RTD*6)-($LST*2)-($FUM*2)+($RECTD*6)+($RECYDS/10);
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
                                $PATATT = round($data[15]);//PAT Att
                                $PAT_Made = round($data[16]);//PAT Made
                                $FGATT = round($data[28]);//FG Att
                                $Long = round($data[29]);//Long Field Goals
                                $Med = round($data[31]);//Med Field Goals
                                $Short = round($data[33]);//Short Field Goals
                                $POINTS = ($Long*5)+($Med*4)+($Short*3)+$PAT_Made-($PATATT-$PAT_Made)
                                        -($PATATT-($Long+$Med+$Short));
                                $array[] = array('points'=>$POINTS, 'name'=>$name, 'team'=>$team, 'PATATT'=>$PATATT,
                                                    'PAT_Made'=>$PAT_Made, 'FGATT'=>$FGATT, 'Long'=>$Long,
                                                    'Med'=>$Med, 'Short'=>$Short);
                            }
                            $count++;
                            if($count >100)
                                break;
                        }
                    }
                    fclose($handle);
                }
                return($array);
            }
            $array = getStats($week_val, $pos, $point);
            function retStats($array, $week_val, $pos, $point)
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
                retStats($array, $week_val, $pos, $point);
        ?>
        <html>
</tr>
<?php require_once("footerT.html"); ?>