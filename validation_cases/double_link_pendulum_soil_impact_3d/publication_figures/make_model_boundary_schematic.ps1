param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot '..\case_config.json'),
    [string]$OutputPath = (Join-Path $PSScriptRoot 'fig05_model_boundary_schematic.svg'),
    [double]$ReleaseAngle1Deg = 150.0,
    [double]$ReleaseAngle2Deg = 170.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$cfg = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json

$degree = [char]0x00B0
$alpha = [char]0x03B1
$omega = [char]0x03C9
$times = [char]0x00D7

function F([double]$value, [string]$format = '0.###') {
    return $value.ToString($format, $culture)
}

function Escape-Xml([string]$value) {
    return [System.Security.SecurityElement]::Escape($value)
}

function P([object[]]$point, [double]$ox, [double]$oy, [double]$scale) {
    if ($null -eq $point -or $point.Count -lt 3) {
        $caller = (Get-PSCallStack | Select-Object -Skip 1 -First 1)
        throw "Invalid 3-D point supplied by $($caller.FunctionName) at line $($caller.ScriptLineNumber): count=$($point.Count) value=$point"
    }
    $x = [double]$point[0]
    $y = [double]$point[1]
    $z = [double]$point[2]
    return [pscustomobject]@{
        X = $ox + $scale * (0.92 * $x - 0.45 * $y)
        Y = $oy + $scale * (0.22 * $x + 0.32 * $y - 0.88 * $z)
    }
}

function Point-String([object[]]$points, [double]$ox, [double]$oy, [double]$scale) {
    $tokens = foreach ($point in $points) {
        $q = P $point $ox $oy $scale
        (F $q.X) + ',' + (F $q.Y)
    }
    return $tokens -join ' '
}

function Add-Polygon(
    [System.Text.StringBuilder]$svg,
    [object[]]$points,
    [double]$ox,
    [double]$oy,
    [double]$scale,
    [string]$fill,
    [string]$stroke = '#333333',
    [double]$strokeWidth = 1.8,
    [string]$extra = ''
) {
    $pts = Point-String $points $ox $oy $scale
    [void]$svg.AppendLine('<polygon points="' + $pts + '" fill="' + $fill + '" stroke="' + $stroke + '" stroke-width="' + (F $strokeWidth) + '" ' + $extra + '/>')
}

function Add-Line3D(
    [System.Text.StringBuilder]$svg,
    [object[]]$a,
    [object[]]$b,
    [double]$ox,
    [double]$oy,
    [double]$scale,
    [string]$class = 'edge',
    [string]$extra = ''
) {
    $pa = P $a $ox $oy $scale
    $pb = P $b $ox $oy $scale
    [void]$svg.AppendLine('<line class="' + $class + '" x1="' + (F $pa.X) + '" y1="' + (F $pa.Y) + '" x2="' + (F $pb.X) + '" y2="' + (F $pb.Y) + '" ' + $extra + '/>')
}

function Add-Text(
    [System.Text.StringBuilder]$svg,
    [double]$x,
    [double]$y,
    [string]$text,
    [string]$class = 'label',
    [string]$anchor = 'start',
    [string]$extra = ''
) {
    [void]$svg.AppendLine('<text class="' + $class + '" x="' + (F $x) + '" y="' + (F $y) + '" text-anchor="' + $anchor + '" ' + $extra + '>' + (Escape-Xml $text) + '</text>')
}

function Add-Leader(
    [System.Text.StringBuilder]$svg,
    [double]$x1,
    [double]$y1,
    [double]$x2,
    [double]$y2,
    [string]$text,
    [string]$anchor = 'start'
) {
    $elbowX = if ($anchor -eq 'end') { $x2 + 18.0 } else { $x2 - 18.0 }
    [void]$svg.AppendLine('<polyline class="leader" points="' + (F $x1) + ',' + (F $y1) + ' ' + (F $elbowX) + ',' + (F $y2) + ' ' + (F $x2) + ',' + (F $y2) + '"/>')
    Add-Text $svg $x2 ($y2 - 7.0) $text 'label' $anchor
}

function Add-WireBox(
    [System.Text.StringBuilder]$svg,
    [object[]]$lo,
    [object[]]$hi,
    [double]$ox,
    [double]$oy,
    [double]$scale,
    [string]$class = 'domain-edge'
) {
    $x0=[double]$lo[0]; $y0=[double]$lo[1]; $z0=[double]$lo[2]
    $x1=[double]$hi[0]; $y1=[double]$hi[1]; $z1=[double]$hi[2]
    $v = @(
        @($x0,$y0,$z0), @($x1,$y0,$z0), @($x1,$y1,$z0), @($x0,$y1,$z0),
        @($x0,$y0,$z1), @($x1,$y0,$z1), @($x1,$y1,$z1), @($x0,$y1,$z1)
    )
    foreach ($edge in @(@(0,1),@(1,2),@(2,3),@(3,0),@(4,5),@(5,6),@(6,7),@(7,4),@(0,4),@(1,5),@(2,6),@(3,7))) {
        Add-Line3D $svg $v[$edge[0]] $v[$edge[1]] $ox $oy $scale $class
    }
}

function Add-MediumBlock(
    [System.Text.StringBuilder]$svg,
    [object[]]$lo,
    [object[]]$hi,
    [double]$ox,
    [double]$oy,
    [double]$scale,
    [switch]$BoundaryInset
) {
    $x0=[double]$lo[0]; $y0=[double]$lo[1]; $z0=[double]$lo[2]
    $x1=[double]$hi[0]; $y1=[double]$hi[1]; $z1=[double]$hi[2]
    $bottom = @(@($x0,$y0,$z0),@($x1,$y0,$z0),@($x1,$y1,$z0),@($x0,$y1,$z0))
    $faceY = @(@($x0,$y1,$z0),@($x1,$y1,$z0),@($x1,$y1,$z1),@($x0,$y1,$z1))
    $faceX = @(@($x0,$y0,$z0),@($x0,$y1,$z0),@($x0,$y1,$z1),@($x0,$y0,$z1))
    $top = @(@($x0,$y0,$z1),@($x1,$y0,$z1),@($x1,$y1,$z1),@($x0,$y1,$z1))
    if ($BoundaryInset) {
        Add-Polygon $svg $bottom $ox $oy $scale 'url(#fixedHatch)' '#555555' 2.1
        Add-Polygon $svg $faceY $ox $oy $scale '#E5F3F8' '#0077BB' 2.2 'fill-opacity="0.72"'
        Add-Polygon $svg $faceX $ox $oy $scale '#F5E6EF' '#CC79A7' 2.2 'fill-opacity="0.72"'
        Add-Polygon $svg $top $ox $oy $scale '#BFE3F2' '#0077BB' 2.2 'fill-opacity="0.68"'
    } else {
        Add-Polygon $svg $bottom $ox $oy $scale '#D7EAF2' '#457487' 1.7 'fill-opacity="0.55"'
        Add-Polygon $svg $faceY $ox $oy $scale '#8BC9E3' '#457487' 1.8 'fill-opacity="0.48"'
        Add-Polygon $svg $faceX $ox $oy $scale '#75B8D5' '#457487' 1.8 'fill-opacity="0.44"'
        Add-Polygon $svg $top $ox $oy $scale '#BFE3F2' '#0077BB' 2.1 'fill-opacity="0.62"'
    }
}

function Add-PrismaticLink(
    [System.Text.StringBuilder]$svg,
    [object[]]$start,
    [object[]]$end,
    [double]$width,
    [double]$thickness,
    [double]$ox,
    [double]$oy,
    [double]$scale,
    [string]$fill
) {
    $dx=[double]$end[0]-[double]$start[0]
    $dz=[double]$end[2]-[double]$start[2]
    $length=[math]::Sqrt($dx*$dx+$dz*$dz)
    $nx=-$dz/$length
    $nz=$dx/$length
    $hw=0.5*$width
    $ht=0.5*$thickness
    $s0=@(([double]$start[0]+$nx*$hw),(-$ht),([double]$start[2]+$nz*$hw))
    $s1=@(([double]$start[0]-$nx*$hw),(-$ht),([double]$start[2]-$nz*$hw))
    $e0=@(([double]$end[0]+$nx*$hw),(-$ht),([double]$end[2]+$nz*$hw))
    $e1=@(([double]$end[0]-$nx*$hw),(-$ht),([double]$end[2]-$nz*$hw))
    $s2=@($s0[0],$ht,$s0[2]); $s3=@($s1[0],$ht,$s1[2])
    $e2=@($e0[0],$ht,$e0[2]); $e3=@($e1[0],$ht,$e1[2])
    Add-Polygon $svg @($s0,$e0,$e1,$s1) $ox $oy $scale '#ECECEC' '#333333' 1.8
    Add-Polygon $svg @($s0,$s2,$e2,$e0) $ox $oy $scale $fill '#333333' 1.9
    Add-Polygon $svg @($s2,$e2,$e3,$s3) $ox $oy $scale $fill '#333333' 2.0 'fill-opacity="0.88"'
}

function Add-CylinderHammer(
    [System.Text.StringBuilder]$svg,
    [object[]]$center,
    [double]$radius,
    [double]$width,
    [double]$ox,
    [double]$oy,
    [double]$scale
) {
    $back=@(); $front=@()
    for($i=0;$i -lt 40;$i++){
        $theta=2.0*[math]::PI*$i/40.0
        $x=[double]$center[0]+$radius*[math]::Cos($theta)
        $z=[double]$center[2]+$radius*[math]::Sin($theta)
        $back += ,@($x,(-0.5*$width),$z)
        $front += ,@($x,(0.5*$width),$z)
    }
    Add-Polygon $svg $back $ox $oy $scale '#F5C6A5' '#8D3F19' 1.8 'fill-opacity="0.92"'
    foreach($i in @(0,5,10,15,20,25,30,35)){
        $a=P $back[$i] $ox $oy $scale
        $b=P $front[$i] $ox $oy $scale
        [void]$svg.AppendLine('<line class="contact-mesh" x1="'+(F $a.X)+'" y1="'+(F $a.Y)+'" x2="'+(F $b.X)+'" y2="'+(F $b.Y)+'"/>')
    }
    Add-Polygon $svg $front $ox $oy $scale '#EE7733' '#8D3F19' 2.2 'fill-opacity="0.93"'
    for($i=0;$i -lt 40;$i+=5){
        $a=P $front[$i] $ox $oy $scale
        $b=P @([double]$center[0],(0.5*$width),[double]$center[2]) $ox $oy $scale
        [void]$svg.AppendLine('<line class="contact-mesh" x1="'+(F $a.X)+'" y1="'+(F $a.Y)+'" x2="'+(F $b.X)+'" y2="'+(F $b.Y)+'"/>')
    }
}

function Add-Joint(
    [System.Text.StringBuilder]$svg,
    [object[]]$point,
    [double]$axisHalfLength,
    [double]$ox,
    [double]$oy,
    [double]$scale,
    [string]$fill
) {
    $a=@([double]$point[0],(-$axisHalfLength),[double]$point[2])
    $b=@([double]$point[0], $axisHalfLength,[double]$point[2])
    Add-Line3D $svg $a $b $ox $oy $scale 'joint-axis'
    foreach($p3 in @($a,$b)){
        $q=P $p3 $ox $oy $scale
        [void]$svg.AppendLine('<circle cx="'+(F $q.X)+'" cy="'+(F $q.Y)+'" r="8" fill="'+$fill+'" stroke="#222222" stroke-width="2"/>')
    }
    $q=P $point $ox $oy $scale
    [void]$svg.AppendLine('<circle cx="'+(F $q.X)+'" cy="'+(F $q.Y)+'" r="7" fill="#FFFFFF" stroke="#222222" stroke-width="2.2"/>')
}

function Add-AngleArc(
    [System.Text.StringBuilder]$svg,
    [object[]]$center,
    [double]$radius,
    [double]$angleDeg,
    [double]$ox,
    [double]$oy,
    [double]$scale
) {
    $path = [System.Text.StringBuilder]::new()
    $segments=[int][math]::Ceiling([math]::Abs($angleDeg)/6.0)
    for($i=0;$i -le $segments;$i++){
        $angle=$angleDeg*$i/$segments*[math]::PI/180.0
        $p3=@(([double]$center[0]+$radius*[math]::Cos($angle)),0.0,([double]$center[2]+$radius*[math]::Sin($angle)))
        $q=P $p3 $ox $oy $scale
        if($i -eq 0){[void]$path.Append('M'+(F $q.X)+','+(F $q.Y))}else{[void]$path.Append(' L'+(F $q.X)+','+(F $q.Y))}
    }
    [void]$svg.AppendLine('<path class="angle-arc" d="'+$path.ToString()+'"/>')
}

function Add-Dimension3D(
    [System.Text.StringBuilder]$svg,
    [object[]]$a,
    [object[]]$b,
    [double]$ox,
    [double]$oy,
    [double]$scale,
    [string]$label,
    [double]$labelDx=0.0,
    [double]$labelDy=-8.0
) {
    $pa=P $a $ox $oy $scale
    $pb=P $b $ox $oy $scale
    [void]$svg.AppendLine('<line class="dimension" x1="'+(F $pa.X)+'" y1="'+(F $pa.Y)+'" x2="'+(F $pb.X)+'" y2="'+(F $pb.Y)+'"/>')
    Add-Text $svg (0.5*($pa.X+$pb.X)+$labelDx) (0.5*($pa.Y+$pb.Y)+$labelDy) $label 'small' 'middle'
}

$pivotX=[double]$cfg.model.pivot[0]
$pivotY=[double]$cfg.model.pivot[1]
$pivotZ=[double]$cfg.model.pivot[2]
$pivot=@($pivotX,$pivotY,$pivotZ)
$a1=$ReleaseAngle1Deg*[math]::PI/180.0
$a2=$ReleaseAngle2Deg*[math]::PI/180.0
$link1Length = [double]$cfg.model.link1_length
$hammerOffset = [double]$cfg.model.hammer_center_from_joint
$jointA=@(
    ([double]$pivot[0] + $link1Length * [math]::Cos($a1)),
    0.0,
    ([double]$pivot[2] + $link1Length * [math]::Sin($a1))
)
$hammerCenter=@(
    ([double]$jointA[0] + $hammerOffset * [math]::Cos($a2)),
    0.0,
    ([double]$jointA[2] + $hammerOffset * [math]::Sin($a2))
)
$soilLo=@([double]$cfg.soil.bounds_lo[0];[double]$cfg.soil.bounds_lo[1];[double]$cfg.soil.bounds_lo[2])
$soilHi=@([double]$cfg.soil.bounds_hi[0];[double]$cfg.soil.bounds_hi[1];[double]$cfg.soil.bounds_hi[2])
$domainLo=@([double]$cfg.soil.domain_lo[0];[double]$cfg.soil.domain_lo[1];[double]$cfg.soil.domain_lo[2])
$domainHi=@([double]$cfg.soil.domain_hi[0];[double]$cfg.soil.domain_hi[1];[double]$cfg.soil.domain_hi[2])
$surfaceZ=[double]$soilHi[2]
$hammerBottom=$hammerCenter[2]-[double]$cfg.model.hammer_radius
$initialClearance=$hammerBottom-$surfaceZ

$svg=[System.Text.StringBuilder]::new()
[void]$svg.AppendLine('<?xml version="1.0" encoding="UTF-8"?>')
[void]$svg.AppendLine('<svg xmlns="http://www.w3.org/2000/svg" width="175mm" height="98.4mm" viewBox="0 0 1600 900" role="img" aria-labelledby="svg-title svg-desc">')
[void]$svg.AppendLine('<title id="svg-title">Three-dimensional double-link pendulum and MPM boundary conditions</title>')
[void]$svg.AppendLine('<desc id="svg-desc">Editable publication schematic of the release configuration with zero angular velocities, the MPM material block, background grid domain, revolute joints, hammer contact surface, and boundary conditions.</desc>')
[void]$svg.AppendLine('<defs>')
[void]$svg.AppendLine('<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#222222"/></marker>')
[void]$svg.AppendLine('<marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#0077BB"/></marker>')
[void]$svg.AppendLine('<marker id="arrowPink" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#CC79A7"/></marker>')
[void]$svg.AppendLine('<marker id="dimStart" markerWidth="8" markerHeight="8" refX="1" refY="3" orient="auto"><path d="M7,0 L7,6 L0,3 z" fill="#555555"/></marker>')
[void]$svg.AppendLine('<marker id="dimEnd" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#555555"/></marker>')
[void]$svg.AppendLine('<pattern id="fixedHatch" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="12" height="12" fill="#EFEFEF"/><line x1="0" y1="0" x2="0" y2="12" stroke="#777777" stroke-width="3"/></pattern>')
[void]$svg.AppendLine('</defs>')
[void]$svg.AppendLine('<rect x="0" y="0" width="1600" height="900" fill="#FFFFFF"/>')
[void]$svg.AppendLine('<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202020;font-weight:400}.panel-label{font-size:27px;font-weight:700}.panel-title{font-size:24px;font-weight:500}.label{font-size:22px}.small{font-size:20px}.note{font-size:19px;fill:#4F4F4F}.edge{stroke:#333333;stroke-width:2;fill:none}.domain-edge{stroke:#8A9AA3;stroke-width:1.7;stroke-dasharray:9 7;fill:none}.joint-axis{stroke:#222222;stroke-width:4.2;stroke-linecap:round}.contact-mesh{stroke:#8D3F19;stroke-width:1.05;opacity:.65}.leader{stroke:#555555;stroke-width:1.7;fill:none}.angle-arc{stroke:#333333;stroke-width:2;stroke-dasharray:5 4;fill:none;marker-end:url(#arrow)}.dimension{stroke:#555555;stroke-width:1.6;fill:none;marker-start:url(#dimStart);marker-end:url(#dimEnd)}.bc-x{stroke:#0077BB;stroke-width:2.5;fill:none;marker-end:url(#arrowBlue)}.bc-y{stroke:#CC79A7;stroke-width:2.5;fill:none;marker-end:url(#arrowPink)}.gravity{stroke:#222222;stroke-width:2.8;fill:none;marker-end:url(#arrow)}</style>')

$ox=820.0; $oy=690.0; $scale=500.0
Add-Text $svg 30 42 '(a)' 'panel-label'
Add-Text $svg 82 42 'Three-dimensional coupled model at release' 'panel-title'

Add-WireBox $svg $domainLo $domainHi $ox $oy $scale 'domain-edge'
Add-MediumBlock $svg $soilLo $soilHi $ox $oy $scale
Add-PrismaticLink $svg $pivot $jointA ([double]$cfg.model.link1_width) ([double]$cfg.model.link1_thickness) $ox $oy $scale '#6F7F87'
Add-PrismaticLink $svg $jointA $hammerCenter ([double]$cfg.model.link2_rod_width) ([double]$cfg.model.link2_rod_thickness) $ox $oy $scale '#9A7B5A'
Add-CylinderHammer $svg $hammerCenter ([double]$cfg.model.hammer_radius) ([double]$cfg.model.hammer_width) $ox $oy $scale
Add-Joint $svg $pivot 0.055 $ox $oy $scale '#0077BB'
Add-Joint $svg $jointA 0.052 $ox $oy $scale '#FFFFFF'

$pO=P $pivot $ox $oy $scale
$pA=P $jointA $ox $oy $scale
$pC=P $hammerCenter $ox $oy $scale
Add-Text $svg ($pO.X+14) ($pO.Y-13) 'O' 'label'
Add-Text $svg ($pA.X+13) ($pA.Y-12) 'A' 'label'
Add-Text $svg ($pC.X-12) ($pC.Y-42) 'C' 'label' 'middle'

$refO=@(($pivot[0]+0.18),0.0,$pivot[2])
$refA=@(($jointA[0]+0.16),0.0,$jointA[2])
Add-Line3D $svg $pivot $refO $ox $oy $scale 'domain-edge'
Add-Line3D $svg $jointA $refA $ox $oy $scale 'domain-edge'
Add-AngleArc $svg $pivot 0.105 $ReleaseAngle1Deg $ox $oy $scale
Add-AngleArc $svg $jointA 0.085 $ReleaseAngle2Deg $ox $oy $scale
$alpha1Label=$alpha+'1 = '+(F $ReleaseAngle1Deg '0')+$degree
$alpha2Label=$alpha+'2 = '+(F $ReleaseAngle2Deg '0')+$degree
Add-Text $svg ($pO.X+15) ($pO.Y+53) $alpha1Label 'small' 'middle'
Add-Text $svg ($pA.X-48) ($pA.Y+62) $alpha2Label 'small' 'middle'

$releaseText='Release state: '+$omega+'1 = '+$omega+'2 = 0'
[void]$svg.AppendLine('<rect x="45" y="69" width="342" height="42" fill="#FFFFFF" stroke="#777777" stroke-width="1.4"/>')
Add-Text $svg 62 98 $releaseText 'label'

$gStart=@(($pivot[0]+0.24),0.0,($pivot[2]+0.12))
$gEnd=@(($pivot[0]+0.24),0.0,($pivot[2]-0.12))
$pg0=P $gStart $ox $oy $scale; $pg1=P $gEnd $ox $oy $scale
[void]$svg.AppendLine('<line class="gravity" x1="'+(F $pg0.X)+'" y1="'+(F $pg0.Y)+'" x2="'+(F $pg1.X)+'" y2="'+(F $pg1.Y)+'"/>')
Add-Text $svg ($pg0.X+18) ($pg0.Y+34) ('g = '+(F ([double]$cfg.model.gravity) '0.00')+' m s^-2') 'small'

$dimOffset1=@(($pivot[0]-0.015),0.09,($pivot[2]+0.025))
$dimOffset2=@(($jointA[0]-0.015),0.09,($jointA[2]+0.02))
$dimEnd1=@(($jointA[0]-0.015),0.09,($jointA[2]+0.025))
$dimEnd2=@(($hammerCenter[0]-0.015),0.09,($hammerCenter[2]+0.02))
Add-Dimension3D $svg $dimOffset1 $dimEnd1 $ox $oy $scale ('L1 = '+(F ([double]$cfg.model.link1_length) '0.000')+' m') 20 -17
Add-Dimension3D $svg $dimOffset2 $dimEnd2 $ox $oy $scale ('L2 = '+(F ([double]$cfg.model.hammer_center_from_joint) '0.000')+' m') 22 -18

$clearX=$hammerCenter[0]-0.11
Add-Dimension3D $svg @($clearX,0.0,$surfaceZ) @($clearX,0.0,$hammerBottom) $ox $oy $scale ('h0 = '+(F $initialClearance '0.000')+' m') -56 0
Add-Dimension3D $svg @(($pivot[0]+0.055),0.0,$surfaceZ) @(($pivot[0]+0.055),0.0,$pivot[2]) $ox $oy $scale ('HO = '+(F $pivot[2] '0.000')+' m') 50 0

$soilTopCenter=@((0.5*($soilLo[0]+$soilHi[0])),0.0,$soilHi[2])
$pSoil=P $soilTopCenter $ox $oy $scale
Add-Text $svg 585 720 'MPM medium (water or sand)' 'label' 'middle'
$pDomain=P @($domainLo[0],$domainLo[1],$domainHi[2]) $ox $oy $scale
[void]$svg.AppendLine('<line class="domain-edge" x1="45" y1="335" x2="112" y2="335"/>')
Add-Text $svg 125 342 'Background grid domain' 'label' 'start'
Add-Leader $svg $pO.X $pO.Y 930 365 'Fixed revolute joint O; axis parallel to y' 'start'
Add-Leader $svg $pA.X $pA.Y 705 195 'Revolute joint A; axis parallel to y' 'start'
Add-Leader $svg $pC.X $pC.Y 45 166 'Rigid hammer: peripheral and end-face contact patches' 'start'

$triadOrigin=@(($soilLo[0]+0.08),($soilLo[1]-0.02),($soilLo[2]-0.01))
$po=P $triadOrigin $ox $oy $scale
$px=P @(($triadOrigin[0]+0.13),$triadOrigin[1],$triadOrigin[2]) $ox $oy $scale
$py=P @($triadOrigin[0],($triadOrigin[1]+0.13),$triadOrigin[2]) $ox $oy $scale
$pz=P @($triadOrigin[0],$triadOrigin[1],($triadOrigin[2]+0.13)) $ox $oy $scale
foreach($axis in @(@($px,'x'),@($py,'y'),@($pz,'z'))){
    $q=$axis[0]
    [void]$svg.AppendLine('<line class="gravity" x1="'+(F $po.X)+'" y1="'+(F $po.Y)+'" x2="'+(F $q.X)+'" y2="'+(F $q.Y)+'"/>')
    Add-Text $svg ($q.X+8) ($q.Y-4) ([string]$axis[1]) 'small'
}

$insetX=1450.0; $insetY=480.0; $insetScale=420.0
Add-Text $svg 1030 42 '(b)' 'panel-label'
Add-Text $svg 1082 42 'MPM boundary conditions' 'panel-title'
Add-MediumBlock $svg $soilLo $soilHi $insetX $insetY $insetScale -BoundaryInset

$topCenter=P @((0.5*($soilLo[0]+$soilHi[0])),0.0,$soilHi[2]) $insetX $insetY $insetScale
Add-Leader $svg $topCenter.X $topCenter.Y 1560 330 'Free surface: z = 0' 'end'
$bottomCenter=P @((0.5*($soilLo[0]+$soilHi[0])),0.0,$soilLo[2]) $insetX $insetY $insetScale
Add-Leader $svg $bottomCenter.X $bottomCenter.Y 1560 635 'Fixed bottom' 'end'
Add-Text $svg 1560 660 'vx = vy = vz = 0' 'small' 'end'

$midZ = 0.5 * ([double]$soilLo[2] + [double]$soilHi[2])
$midX = 0.5 * ([double]$soilLo[0] + [double]$soilHi[0])
$xFaceCenter=@([double]$soilLo[0];0.0;$midZ)
$pxc=P $xFaceCenter $insetX $insetY $insetScale
[void]$svg.AppendLine('<line class="bc-x" x1="'+(F ($pxc.X-44))+'" y1="'+(F ($pxc.Y+10))+'" x2="'+(F ($pxc.X-4))+'" y2="'+(F $pxc.Y)+'"/>')
Add-Text $svg 1035 628 'x-normal rollers: vx = 0' 'small' 'start'

$yFaceCenter=@($midX;[double]$soilHi[1];$midZ)
$pyc=P $yFaceCenter $insetX $insetY $insetScale
[void]$svg.AppendLine('<line class="bc-y" x1="'+(F ($pyc.X+30))+'" y1="'+(F ($pyc.Y+30))+'" x2="'+(F ($pyc.X+3))+'" y2="'+(F ($pyc.Y+3))+'"/>')
Add-Text $svg 1208 700 'y-normal rollers: vy = 0' 'small' 'start'

$mediumDims=(F ($soilHi[0]-$soilLo[0]) '0.00')+' '+$times+' '+(F ($soilHi[1]-$soilLo[1]) '0.00')+' '+$times+' '+(F ($soilHi[2]-$soilLo[2]) '0.00')+' m'
Add-Text $svg 1300 770 ('MPM material region: '+$mediumDims) 'note' 'middle'
$domainDims=(F ($domainHi[0]-$domainLo[0]) '0.00')+' '+$times+' '+(F ($domainHi[1]-$domainLo[1]) '0.00')+' '+$times+' '+(F ($domainHi[2]-$domainLo[2]) '0.00')+' m'
Add-Text $svg 1300 806 ('Background grid: '+$domainDims) 'note' 'middle'
Add-Text $svg 800 872 'Motion is constrained to the x-z plane; both revolute-joint axes are parallel to y.' 'note' 'middle'

[void]$svg.AppendLine('</svg>')
[System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($OutputPath)) | Out-Null
[System.IO.File]::WriteAllText($OutputPath,$svg.ToString(),[System.Text.UTF8Encoding]::new($false))
Write-Output "Generated $OutputPath"
