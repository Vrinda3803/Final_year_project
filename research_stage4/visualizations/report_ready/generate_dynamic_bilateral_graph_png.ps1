Add-Type -AssemblyName System.Drawing
$out=Join-Path $PSScriptRoot 'dynamic_bilateral_graph_transparent.png'
$bmp=[System.Drawing.Bitmap]::new(512,512,[System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g=[System.Drawing.Graphics]::FromImage($bmp);$g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias;$g.Clear([System.Drawing.Color]::Transparent)
function C([string]$h){[System.Drawing.ColorTranslator]::FromHtml($h)}
function Line([float]$x1,[float]$y1,[float]$x2,[float]$y2,[string]$color,[float]$width){$p=[System.Drawing.Pen]::new((C $color),$width);$p.StartCap=[System.Drawing.Drawing2D.LineCap]::Round;$p.EndCap=[System.Drawing.Drawing2D.LineCap]::Round;$g.DrawLine($p,$x1,$y1,$x2,$y2);$p.Dispose()}
function ArrowHead([float]$x,[float]$y,[float]$angle,[string]$color){$len=23;$spread=0.58;$x1=$x-$len*[Math]::Cos($angle-$spread);$y1=$y-$len*[Math]::Sin($angle-$spread);$x2=$x-$len*[Math]::Cos($angle+$spread);$y2=$y-$len*[Math]::Sin($angle+$spread);Line $x $y $x1 $y1 $color 7;Line $x $y $x2 $y2 $color 7}
# Anatomical BODY_25-style links.
$bone=[System.Drawing.Pen]::new((C '#68717B'),11);$bone.StartCap=[System.Drawing.Drawing2D.LineCap]::Round;$bone.EndCap=[System.Drawing.Drawing2D.LineCap]::Round
foreach($s in @(@(256,57,158,151),@(158,151,80,266),@(80,266,164,365),@(164,365,104,466),@(256,57,354,151),@(354,151,432,266),@(432,266,348,365),@(348,365,408,466))){$g.DrawLine($bone,$s[0],$s[1],$s[2],$s[3])};$bone.Dispose()
# Time-varying bilateral links: curved connections with arrowheads at both ends.
$arc=[System.Drawing.Pen]::new((C '#D9825B'),9);$arc.StartCap=[System.Drawing.Drawing2D.LineCap]::Round;$arc.EndCap=[System.Drawing.Drawing2D.LineCap]::Round
$g.DrawBezier($arc,186,151,220,105,292,105,326,151)
$g.DrawBezier($arc,108,266,180,312,332,312,404,266)
$arc.Dispose();ArrowHead 216 132 (-0.42) '#D9825B';ArrowHead 296 132 ([Math]::PI+0.42) '#D9825B';ArrowHead 160 284 0.30 '#D9825B';ArrowHead 352 284 ([Math]::PI-0.30) '#D9825B'
# Joint nodes.
$nodes=@(@(256,57,24,'#D96F82'),@(158,151,24,'#D96F82'),@(354,151,24,'#D96F82'),@(80,266,24,'#4D91C1'),@(432,266,24,'#4D91C1'),@(164,365,24,'#62A879'),@(348,365,24,'#62A879'),@(104,466,18,'#62A879'),@(408,466,18,'#62A879'))
foreach($n in $nodes){$b=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::White);$p=[System.Drawing.Pen]::new((C $n[3]),12);$g.FillEllipse($b,$n[0]-$n[2],$n[1]-$n[2],2*$n[2],2*$n[2]);$g.DrawEllipse($p,$n[0]-$n[2],$n[1]-$n[2],2*$n[2],2*$n[2]);$b.Dispose();$p.Dispose()}
$bmp.Save($out,[System.Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$bmp.Dispose();Write-Output $out
