Add-Type -AssemblyName System.Drawing
$out = Join-Path $PSScriptRoot 'pose_sequence_frame_stack.png'
$bmp = [System.Drawing.Bitmap]::new(512,512,[System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.Clear([System.Drawing.Color]::Transparent)
function C([string]$h){[System.Drawing.ColorTranslator]::FromHtml($h)}
function DrawPosePanel([float]$tx,[float]$ty){
    $state=$g.Save()
    $matrix=[System.Drawing.Drawing2D.Matrix]::new(1,0,-0.21,1,$tx,$ty)
    $g.Transform=$matrix
    $fill=[System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255,5,5,5))
    $border=[System.Drawing.Pen]::new((C '#B8CDE0'),4)
    $g.FillRectangle($fill,0,0,150,330);$g.DrawRectangle($border,0,0,150,330)
    $fill.Dispose();$border.Dispose();$matrix.Dispose()
    $bones=@(
        @(75,44,75,90,'#F2D52A'),@(75,90,49,103,'#55D98A'),@(49,103,38,147,'#55D98A'),@(38,147,31,192,'#17C5D5'),
        @(75,90,101,103,'#E4D42A'),@(101,103,112,147,'#9BD832'),@(112,147,119,192,'#E4D42A'),@(49,103,101,103,'#B9DE2D'),
        @(75,90,75,185,'#25D0C2'),@(75,185,57,251,'#258CE7'),@(57,251,52,304,'#873BD6'),@(75,185,94,251,'#B53BD0'),@(94,251,99,304,'#D637CF')
    )
    foreach($b in $bones){$p=[System.Drawing.Pen]::new((C $b[4]),8);$p.StartCap=[System.Drawing.Drawing2D.LineCap]::Round;$p.EndCap=[System.Drawing.Drawing2D.LineCap]::Round;$g.DrawLine($p,$b[0],$b[1],$b[2],$b[3]);$p.Dispose()}
    $joints=@(@(75,31,10,'#F04474'),@(75,90,7,'#F2D52A'),@(49,103,7,'#B9DE2D'),@(101,103,7,'#E4D42A'),@(38,147,7,'#55D98A'),@(112,147,7,'#9BD832'),@(31,192,7,'#17C5D5'),@(119,192,7,'#E4D42A'),@(75,185,7,'#25D0C2'),@(57,251,7,'#258CE7'),@(94,251,7,'#A43BD0'),@(52,304,7,'#873BD6'),@(99,304,7,'#D637CF'))
    foreach($j in $joints){$b=[System.Drawing.SolidBrush]::new((C $j[3]));$p=[System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(255,17,17,17),3);$g.FillEllipse($b,$j[0]-$j[2],$j[1]-$j[2],2*$j[2],2*$j[2]);$g.DrawEllipse($p,$j[0]-$j[2],$j[1]-$j[2],2*$j[2],2*$j[2]);$b.Dispose();$p.Dispose()}
    $g.Restore($state)
}
DrawPosePanel 74 72
DrawPosePanel 184 39
DrawPosePanel 294 6
$bmp.Save($out,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
Write-Output $out
