Add-Type -AssemblyName System.Drawing
$dir=$PSScriptRoot
$ink=[System.Drawing.Color]::FromArgb(255,52,59,74)
function New-Icon([string]$name,[scriptblock]$draw){$bmp=[System.Drawing.Bitmap]::new(256,256,[System.Drawing.Imaging.PixelFormat]::Format32bppArgb);$g=[System.Drawing.Graphics]::FromImage($bmp);$g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias;$g.Clear([System.Drawing.Color]::Transparent);& $draw $g;$bmp.Save((Join-Path $dir $name),[System.Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$bmp.Dispose()}
function Make-Pen([float]$w){$p=[System.Drawing.Pen]::new($ink,$w);$p.StartCap=[System.Drawing.Drawing2D.LineCap]::Round;$p.EndCap=[System.Drawing.Drawing2D.LineCap]::Round;$p.LineJoin=[System.Drawing.Drawing2D.LineJoin]::Round;return $p}

New-Icon 'preprocessing_gear_transparent.png' {
    param($g)
    $points=[System.Drawing.PointF[]]::new(32)
    for($i=0;$i -lt 32;$i++){$a=-[Math]::PI/2+$i*2*[Math]::PI/32;$r=if(($i%4) -in @(1,2)){108}else{91};$points[$i]=[System.Drawing.PointF]::new([float](128+$r*[Math]::Cos($a)),[float](128+$r*[Math]::Sin($a)))}
    $p=Make-Pen 11;$g.DrawPolygon($p,$points);$g.DrawEllipse($p,91,91,74,74);$p.Dispose()
}
New-Icon 'evaluation_database_transparent.png' {
    param($g)
    $p=Make-Pen 11
    $g.DrawEllipse($p,45,25,166,58);$g.DrawLine($p,45,54,45,196);$g.DrawLine($p,211,54,211,196)
    $g.DrawArc($p,45,73,166,58,0,180);$g.DrawArc($p,45,121,166,58,0,180);$g.DrawArc($p,45,167,166,58,0,180)
    $g.DrawArc($p,45,25,166,58,180,180)
    $p.Dispose()
}
New-Icon 'data_sliders_transparent.png' {
    param($g)
    $p=Make-Pen 11;$g.DrawLine($p,28,64,228,64);$g.DrawLine($p,28,128,228,128);$g.DrawLine($p,28,192,228,192)
    foreach($n in @(@(84,64),@(166,128),@(112,192))){$b=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::White);$g.FillEllipse($b,$n[0]-18,$n[1]-18,36,36);$g.DrawEllipse($p,$n[0]-18,$n[1]-18,36,36);$b.Dispose()};$p.Dispose()
}
