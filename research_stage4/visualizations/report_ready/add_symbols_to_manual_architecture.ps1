Add-Type -AssemblyName System.Drawing

$source = 'D:\Downloads\system_architecture.png'
$target = Join-Path $PSScriptRoot 'system_architecture_with_reference_symbols.png'
$bmp = [System.Drawing.Bitmap]::FromFile($source)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

$navy = '#253D54'; $blue = '#4D91C1'; $teal = '#4E9C92'
$orange = '#E39B56'; $pink = '#D96F82'; $green = '#62A879'
$pale = '#E8E2C8'

function Get-Color([string]$hex) { [System.Drawing.ColorTranslator]::FromHtml($hex) }
function Draw-Line([float]$x1,[float]$y1,[float]$x2,[float]$y2,[string]$color=$navy,[float]$width=4) {
    $p = [System.Drawing.Pen]::new((Get-Color $color), $width)
    $p.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
    $p.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
    $g.DrawLine($p,$x1,$y1,$x2,$y2); $p.Dispose()
}
function Draw-Circle([float]$cx,[float]$cy,[float]$r,[string]$stroke=$navy,[string]$fill='#FFFFFF',[float]$width=3) {
    $b = [System.Drawing.SolidBrush]::new((Get-Color $fill))
    $p = [System.Drawing.Pen]::new((Get-Color $stroke),$width)
    $g.FillEllipse($b,$cx-$r,$cy-$r,2*$r,2*$r); $g.DrawEllipse($p,$cx-$r,$cy-$r,2*$r,2*$r)
    $b.Dispose(); $p.Dispose()
}
function Draw-Rect([float]$x,[float]$y,[float]$w,[float]$h,[string]$stroke=$navy,[string]$fill='#FFFFFF',[float]$width=3) {
    $b = [System.Drawing.SolidBrush]::new((Get-Color $fill)); $p=[System.Drawing.Pen]::new((Get-Color $stroke),$width)
    $g.FillRectangle($b,$x,$y,$w,$h); $g.DrawRectangle($p,$x,$y,$w,$h); $b.Dispose();$p.Dispose()
}

# 1. Pose skeleton symbol in the Input Representation module.
$pose=@(@(672,154),@(672,180),@(672,203),@(653,191),@(690,191),@(646,211),@(698,211),@(660,230),@(684,230),@(655,254),@(690,254))
$bones=@(@(0,1),@(1,2),@(1,3),@(1,4),@(3,5),@(4,6),@(2,7),@(2,8),@(7,9),@(8,10))
foreach($b in $bones){$a=$pose[$b[0]];$c=$pose[$b[1]];Draw-Line $a[0] $a[1] $c[0] $c[1] $blue 3}
for($i=0;$i -lt $pose.Count;$i++){$col=if($i -eq 0){$pink}elseif($i -in @(3,4,5,6)){$green}else{$orange};Draw-Circle $pose[$i][0] $pose[$i][1] 4 $navy $col 2}

# 2. Gear, database and sliders beside the data-preparation steps.
$gx=1093;$gy=151
for($a=0;$a -lt 8;$a++){$r=[Math]::PI*$a/4;$x1=$gx+[Math]::Cos($r)*10;$y1=$gy+[Math]::Sin($r)*10;$x2=$gx+[Math]::Cos($r)*15;$y2=$gy+[Math]::Sin($r)*15;Draw-Line $x1 $y1 $x2 $y2 $navy 3}
Draw-Circle $gx $gy 10 $navy '#FFFFFF' 3;Draw-Circle $gx $gy 4 $blue '#FFFFFF' 2
$dx=1093;$dy=199
$p=[System.Drawing.Pen]::new((Get-Color $navy),3);$g.DrawEllipse($p,$dx-13,$dy-8,26,10);$g.DrawLine($p,$dx-13,$dy-3,$dx-13,$dy+14);$g.DrawLine($p,$dx+13,$dy-3,$dx+13,$dy+14);$g.DrawArc($p,$dx-13,$dy+7,26,10,0,180);$p.Dispose();Draw-Line ($dx-13) ($dy+6) ($dx+13) ($dy+6) $blue 2
foreach($row in @(@(1080,244,$blue),@(1087,249,$green),@(1083,254,$orange))){Draw-Line 1072 $row[1] 1090 $row[1] $navy 2;Draw-Circle $row[0] $row[1] 2.5 $row[2] '#FFFFFF' 1.5}

# 3. Small body-graph symbol in the RCDB-STGT model module.
$nodes=@(@(1698,151),@(1698,174),@(1680,189),@(1716,189),@(1673,210),@(1689,210),@(1707,210),@(1723,210),@(1685,232),@(1711,232))
$edges=@(@(0,1),@(1,2),@(1,3),@(2,4),@(2,5),@(3,6),@(3,7),@(1,5),@(1,6),@(5,8),@(6,9))
foreach($e in $edges){$a=$nodes[$e[0]];$c=$nodes[$e[1]];Draw-Line $a[0] $a[1] $c[0] $c[1] $blue 2.5}
for($i=0;$i -lt $nodes.Count;$i++){$fill=if($i -eq 0){$pink}else{'#FFFFFF'};Draw-Circle $nodes[$i][0] $nodes[$i][1] 4.5 $navy $fill 2.5}

# 4. Walking/activity figure in the Output module.
Draw-Circle 2049 164 7 $navy '#FFFFFF' 3
Draw-Line 2049 172 2049 198 $blue 4
Draw-Line 2049 179 2034 192 $blue 4;Draw-Line 2049 179 2065 169 $blue 4
Draw-Line 2049 198 2035 219 $blue 4;Draw-Line 2049 198 2066 216 $blue 4
Draw-Circle 2034 192 2.5 $green $green 1;Draw-Circle 2065 169 2.5 $orange $orange 1

# 5. Frame-sequence/transformer symbol in the Temporal Transformer module.
foreach($fr in @(@(1708,512,28,58,$blue),@(1747,505,30,72,'#9276BA'),@(1788,512,28,58,$blue))){
    Draw-Rect $fr[0] $fr[1] $fr[2] $fr[3] $navy '#FFFFFF' 2.5
}
Draw-Line 1737 541 1745 541 $green 3;Draw-Line 1778 541 1786 541 $green 3
$tri=[System.Drawing.PointF[]]@([System.Drawing.PointF]::new(1758,528),[System.Drawing.PointF]::new(1770,541),[System.Drawing.PointF]::new(1758,554))
$tb=[System.Drawing.SolidBrush]::new((Get-Color '#F1EBF8'));$tp=[System.Drawing.Pen]::new((Get-Color '#9276BA'),2);$g.FillPolygon($tb,$tri);$g.DrawPolygon($tp,$tri);$tb.Dispose();$tp.Dispose()

$bmp.Save($target,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
Write-Output $target
