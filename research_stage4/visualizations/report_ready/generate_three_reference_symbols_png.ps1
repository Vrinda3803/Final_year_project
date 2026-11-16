Add-Type -AssemblyName System.Drawing
$dir=$PSScriptRoot
$ink=[System.Drawing.Color]::FromArgb(255,52,59,74)
function New-Canvas([string]$path,[scriptblock]$draw){
    $bmp=[System.Drawing.Bitmap]::new(256,256,[System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g=[System.Drawing.Graphics]::FromImage($bmp);$g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias;$g.Clear([System.Drawing.Color]::Transparent)
    & $draw $g
    $bmp.Save((Join-Path $dir $path),[System.Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$bmp.Dispose()
}
function Pen($color,[float]$width){$p=[System.Drawing.Pen]::new($color,$width);$p.StartCap=[System.Drawing.Drawing2D.LineCap]::Round;$p.EndCap=[System.Drawing.Drawing2D.LineCap]::Round;$p.LineJoin=[System.Drawing.Drawing2D.LineJoin]::Round;return $p}
function Draw-Circle($g,[float]$cx,[float]$cy,[float]$r,[bool]$filled=$false){$p=Pen $ink 11;$g.DrawEllipse($p,$cx-$r,$cy-$r,2*$r,2*$r);if($filled){$b=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::White);$g.FillEllipse($b,$cx-$r+2,$cy-$r+2,2*$r-4,2*$r-4);$b.Dispose()};$p.Dispose()}

# Running person.
New-Canvas 'activity_running_transparent.png' {
    param($g)
    $p=Pen $ink 11
    $g.DrawEllipse($p,130,26,34,34)
    $g.DrawLine($p,135,68,112,126);$g.DrawLine($p,112,126,156,148)
    $g.DrawLine($p,128,82,91,91);$g.DrawLine($p,91,91,67,121);$g.DrawLine($p,67,121,43,128)
    $g.DrawLine($p,133,78,169,67);$g.DrawLine($p,169,67,195,91)
    $g.DrawLine($p,112,126,76,169);$g.DrawLine($p,76,169,49,205)
    $g.DrawLine($p,156,148,191,160);$g.DrawLine($p,191,160,218,198)
    $p.Dispose()
}

# Spatial graph-attention symbol.
New-Canvas 'spatial_graph_attention_transparent.png' {
    param($g)
    $pts=@(@(128,35),@(55,78),@(201,78),@(55,166),@(201,166),@(128,211),@(128,122))
    $edges=@(@(0,1),@(0,2),@(1,3),@(2,4),@(3,5),@(4,5),@(0,6),@(1,6),@(2,6),@(3,6),@(4,6),@(5,6))
    $p=Pen $ink 8
    foreach($e in $edges){$a=$pts[$e[0]];$b=$pts[$e[1]];$g.DrawLine($p,$a[0],$a[1],$b[0],$b[1])}
    $p.Dispose()
    foreach($n in $pts){$br=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::White);$pn=Pen $ink 8;$r=if($n[0] -eq 128 -and $n[1] -eq 122){18}else{15};$g.FillEllipse($br,$n[0]-$r,$n[1]-$r,2*$r,2*$r);$g.DrawEllipse($pn,$n[0]-$r,$n[1]-$r,2*$r,2*$r);$br.Dispose();$pn.Dispose()}
}

# Temporal/video symbol.
New-Canvas 'temporal_video_transparent.png' {
    param($g)
    $p=Pen $ink 9
    $g.DrawRectangle($p,39,69,147,125)
    $g.DrawLine($p,57,51,204,51);$g.DrawLine($p,204,51,204,176)
    $triangle=[System.Drawing.PointF[]]@([System.Drawing.PointF]::new(104,96),[System.Drawing.PointF]::new(104,151),[System.Drawing.PointF]::new(151,123))
    $br=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::White);$g.FillPolygon($br,$triangle);$g.DrawPolygon($p,$triangle);$br.Dispose();$p.Dispose()
}
Get-ChildItem $dir -Filter '*_transparent.png' | Where-Object Name -in @('activity_running_transparent.png','spatial_graph_attention_transparent.png','temporal_video_transparent.png') | Select-Object Name,Length
