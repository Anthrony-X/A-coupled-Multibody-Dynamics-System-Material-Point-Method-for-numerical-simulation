param(
    [string]$WaterCsv = (Join-Path $PSScriptRoot '..\outputs\diagnostics_water.csv'),
    [string]$SandCsv = (Join-Path $PSScriptRoot '..\outputs\diagnostics_sand.csv'),
    [string]$OutputDirectory = $PSScriptRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$culture = [System.Globalization.CultureInfo]::InvariantCulture

$waterColor = '#0072B2'
$sandColor = '#D55E00'
$black = '#202020'
$grey = '#6B6B6B'
$lightGrey = '#D9D9D9'
$thresholdColor = '#7A7A7A'
$quantityColors = @('#0072B2', '#56B4E9', '#D55E00', '#CC79A7')

function F([double]$value, [string]$format = '0.###') {
    return $value.ToString($format, $culture)
}

function Escape-Xml([string]$value) {
    return [System.Security.SecurityElement]::Escape($value)
}

function Get-PropertyRange([object[]]$samples, [string[]]$properties) {
    $minimum = [double]::PositiveInfinity
    $maximum = [double]::NegativeInfinity
    foreach ($sample in $samples) {
        foreach ($property in $properties) {
            $value = [double]$sample.$property
            if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) { continue }
            if ($value -lt $minimum) { $minimum = $value }
            if ($value -gt $maximum) { $maximum = $value }
        }
    }
    if ([double]::IsInfinity($minimum) -or [double]::IsInfinity($maximum)) {
        throw 'Unable to determine a finite plotting range.'
    }
    return [pscustomobject]@{ Minimum = $minimum; Maximum = $maximum }
}

function Get-NiceScale([double]$minimum, [double]$maximum, [int]$targetIntervals = 5, [switch]$ZeroBaseline) {
    if ($ZeroBaseline) { $minimum = [math]::Min(0.0, $minimum) }
    if ($maximum -le $minimum) { $maximum = $minimum + 1.0 }
    $roughStep = ($maximum - $minimum) / [math]::Max(2, $targetIntervals)
    $magnitude = [math]::Pow(10.0, [math]::Floor([math]::Log10($roughStep)))
    $fraction = $roughStep / $magnitude
    $factor = if ($fraction -le 1.0) { 1.0 } elseif ($fraction -le 2.0) { 2.0 } elseif ($fraction -le 2.5) { 2.5 } elseif ($fraction -le 5.0) { 5.0 } else { 10.0 }
    $step = $factor * $magnitude
    $niceMinimum = $step * [math]::Floor($minimum / $step)
    $niceMaximum = $step * [math]::Ceiling($maximum / $step)
    if ($ZeroBaseline) { $niceMinimum = 0.0 }
    if ($niceMaximum -le $niceMinimum) { $niceMaximum = $niceMinimum + $step }
    $count = [int][math]::Round(($niceMaximum - $niceMinimum) / $step)
    $ticks = @(for ($index = 0; $index -le $count; $index++) { $niceMinimum + $index * $step })
    return [pscustomobject]@{ Minimum = $niceMinimum; Maximum = $niceMaximum; Step = $step; Ticks = $ticks }
}

function Read-Diagnostics([string]$path, [string]$name, [string]$color) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing diagnostics file: $path"
    }
    $raw = Import-Csv -LiteralPath $path
    if ($raw.Count -eq 0) {
        throw "Diagnostics file is empty: $path"
    }

    $initialMbd = [double]$raw[0].mbd_mechanical_energy_J
    $initialMedium = (
        [double]$raw[0].soil_kinetic_energy_J +
        [double]$raw[0].soil_potential_energy_J +
        [double]$raw[0].soil_internal_work_cumulative_J
    )
    $samples = foreach ($row in $raw) {
        $theta1 = [double]$row.theta1_rad
        $theta2 = [double]$row.theta2_absolute_rad
        $bottomZ = 0.650 + 0.450 * [math]::Sin($theta1) + 0.350 * [math]::Sin($theta2) - 0.050
        $mediumEnergy = (
            [double]$row.soil_kinetic_energy_J +
            [double]$row.soil_potential_energy_J +
            [double]$row.soil_internal_work_cumulative_J
        )
        [pscustomobject]@{
            TimeMs = 1000.0 * [double]$row.time_s
            Contact = ([int]$row.contact_particles -gt 0)
            PenetrationMm = 1000.0 * [math]::Max(0.0, -$bottomZ)
            BodyForceZkN = [double]$row.contact_force_z_N / 1000.0
            OppositeMediumForceZkN = -[double]$row.soil_contact_force_z_N / 1000.0
            ActionRelative = [double]$row.action_reaction_relative_residual
            GeneralizedAbsoluteW = [double]$row.generalized_power_absolute_residual_W
            GeneralizedRelative = [double]$row.generalized_power_relative_residual
            PowerScaleW = [math]::Max(
                [math]::Abs([double]$row.patch_contact_power_W),
                [math]::Abs([double]$row.generalized_contact_power_W)
            )
            DeltaMbdJ = [double]$row.mbd_mechanical_energy_J - $initialMbd
            MbdWorkJ = [double]$row.mbd_contact_work_cumulative_J + [double]$row.joint_damping_work_cumulative_J
            DeltaMediumJ = $mediumEnergy - $initialMedium
            MediumWorkJ = [double]$row.mpm_soil_contact_work_cumulative_J
            MbdEnergyRelative = [double]$row.mbd_energy_relative_residual
            TotalEnergyRelative = [double]$row.total_energy_relative_residual
        }
    }
    $contact = @($samples | Where-Object Contact)
    $sumError2 = 0.0
    $sumScale2 = 0.0
    foreach ($sample in $contact) {
        $sumError2 += $sample.GeneralizedAbsoluteW * $sample.GeneralizedAbsoluteW
        $sumScale2 += $sample.PowerScaleW * $sample.PowerScaleW
    }
    $globalL2 = if ($sumScale2 -gt 0.0) { [math]::Sqrt($sumError2 / $sumScale2) } else { 0.0 }
    return [pscustomobject]@{
        Name = $name
        Color = $color
        Samples = @($samples)
        ContactSamples = $contact
        GlobalPowerL2 = $globalL2
    }
}

function New-Svg([double]$widthMm, [double]$heightMm, [int]$viewWidth, [int]$viewHeight, [string]$accessibleTitle) {
    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.AppendLine('<?xml version="1.0" encoding="UTF-8"?>')
    [void]$builder.AppendLine((
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}mm" height="{1}mm" viewBox="0 0 {2} {3}" role="img" aria-labelledby="svg-title svg-desc">' -f
        (F $widthMm '0.##'), (F $heightMm '0.##'), $viewWidth, $viewHeight
    ))
    [void]$builder.AppendLine('<title id="svg-title">' + (Escape-Xml $accessibleTitle) + '</title>')
    [void]$builder.AppendLine('<desc id="svg-desc">Editable vector figure generated from the water and sand diagnostics tables.</desc>')
    [void]$builder.AppendLine('<rect x="0" y="0" width="100%" height="100%" fill="#FFFFFF"/>')
    [void]$builder.AppendLine('<style>')
    [void]$builder.AppendLine('text{font-family:Arial,Helvetica,sans-serif;fill:#202020;font-weight:400} .tick{font-size:16px} .axis-label{font-size:20px} .panel-label{font-size:22px;font-weight:700} .panel-title{font-size:19px;font-weight:500} .legend{font-size:16px} .annotation{font-size:15px} .axis{stroke:#202020;stroke-width:1.5;shape-rendering:geometricPrecision} .grid{stroke:#D9D9D9;stroke-width:1} .threshold{stroke:#7A7A7A;stroke-width:1.5;stroke-dasharray:8 6} .series{fill:none;stroke-width:2.6;stroke-linejoin:round;stroke-linecap:round}</style>')
    return $builder
}

function Close-Svg([System.Text.StringBuilder]$builder, [string]$path) {
    [void]$builder.AppendLine('</svg>')
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($path)) | Out-Null
    [System.IO.File]::WriteAllText($path, $builder.ToString(), [System.Text.UTF8Encoding]::new($false))
}

function Add-Panel(
    [System.Text.StringBuilder]$builder,
    [string]$id,
    [double]$x,
    [double]$y,
    [double]$width,
    [double]$height,
    [double]$xMin,
    [double]$xMax,
    [double]$yMin,
    [double]$yMax,
    [double[]]$xTicks,
    [double[]]$yTicks,
    [string]$xLabel,
    [string]$yLabel,
    [string]$panelLabel,
    [string]$panelTitle,
    [ValidateSet('linear','log')][string]$yScale = 'linear',
    [scriptblock]$xFormatter = { param($v) (F $v '0') },
    [scriptblock]$yFormatter = { param($v) (F $v '0.##') }
) {
    $leftMargin = 92.0
    $rightMargin = 24.0
    $topMargin = 54.0
    $bottomMargin = 70.0
    $plotX = $x + $leftMargin
    $plotY = $y + $topMargin
    $plotW = $width - $leftMargin - $rightMargin
    $plotH = $height - $topMargin - $bottomMargin
    $clipId = 'clip-' + $id
    [void]$builder.AppendLine('<defs><clipPath id="' + $clipId + '"><rect x="' + (F $plotX) + '" y="' + (F $plotY) + '" width="' + (F $plotW) + '" height="' + (F $plotH) + '"/></clipPath></defs>')

    foreach ($tick in $yTicks) {
        if ($yScale -eq 'log') {
            $py = $plotY + $plotH - (([math]::Log10($tick) - [math]::Log10($yMin)) / ([math]::Log10($yMax) - [math]::Log10($yMin))) * $plotH
        } else {
            $py = $plotY + $plotH - (($tick - $yMin) / ($yMax - $yMin)) * $plotH
        }
        [void]$builder.AppendLine('<line class="grid" x1="' + (F $plotX) + '" y1="' + (F $py) + '" x2="' + (F ($plotX + $plotW)) + '" y2="' + (F $py) + '"/>')
        [void]$builder.AppendLine('<text class="tick" x="' + (F ($plotX - 13)) + '" y="' + (F ($py + 5)) + '" text-anchor="end">' + (Escape-Xml (& $yFormatter $tick)) + '</text>')
    }
    foreach ($tick in $xTicks) {
        $px = $plotX + (($tick - $xMin) / ($xMax - $xMin)) * $plotW
        [void]$builder.AppendLine('<line class="axis" x1="' + (F $px) + '" y1="' + (F ($plotY + $plotH)) + '" x2="' + (F $px) + '" y2="' + (F ($plotY + $plotH + 7)) + '"/>')
        [void]$builder.AppendLine('<text class="tick" x="' + (F $px) + '" y="' + (F ($plotY + $plotH + 29)) + '" text-anchor="middle">' + (Escape-Xml (& $xFormatter $tick)) + '</text>')
    }
    [void]$builder.AppendLine('<line class="axis" x1="' + (F $plotX) + '" y1="' + (F $plotY) + '" x2="' + (F $plotX) + '" y2="' + (F ($plotY + $plotH)) + '"/>')
    [void]$builder.AppendLine('<line class="axis" x1="' + (F $plotX) + '" y1="' + (F ($plotY + $plotH)) + '" x2="' + (F ($plotX + $plotW)) + '" y2="' + (F ($plotY + $plotH)) + '"/>')
    [void]$builder.AppendLine('<text class="panel-label" x="' + (F ($x + 8)) + '" y="' + (F ($y + 27)) + '">' + (Escape-Xml $panelLabel) + '</text>')
    [void]$builder.AppendLine('<text class="panel-title" x="' + (F ($plotX + 0.5 * $plotW)) + '" y="' + (F ($y + 28)) + '" text-anchor="middle">' + (Escape-Xml $panelTitle) + '</text>')
    [void]$builder.AppendLine('<text class="axis-label" x="' + (F ($plotX + 0.5 * $plotW)) + '" y="' + (F ($y + $height - 10)) + '" text-anchor="middle">' + (Escape-Xml $xLabel) + '</text>')
    [void]$builder.AppendLine('<text class="axis-label" transform="translate(' + (F ($x + 24)) + ',' + (F ($plotY + 0.5 * $plotH)) + ') rotate(-90)" text-anchor="middle">' + (Escape-Xml $yLabel) + '</text>')
    return [pscustomobject]@{
        PlotX = $plotX; PlotY = $plotY; PlotW = $plotW; PlotH = $plotH
        XMin = $xMin; XMax = $xMax; YMin = $yMin; YMax = $yMax
        YScale = $yScale; ClipId = $clipId
    }
}

function Map-X($panel, [double]$value) {
    return $panel.PlotX + (($value - $panel.XMin) / ($panel.XMax - $panel.XMin)) * $panel.PlotW
}

function Map-Y($panel, [double]$value) {
    if ($panel.YScale -eq 'log') {
        return $panel.PlotY + $panel.PlotH - (([math]::Log10($value) - [math]::Log10($panel.YMin)) / ([math]::Log10($panel.YMax) - [math]::Log10($panel.YMin))) * $panel.PlotH
    }
    return $panel.PlotY + $panel.PlotH - (($value - $panel.YMin) / ($panel.YMax - $panel.YMin)) * $panel.PlotH
}

function Add-Series(
    [System.Text.StringBuilder]$builder,
    $panel,
    [object[]]$samples,
    [string]$property,
    [string]$color,
    [string]$dash = '',
    [double]$width = 2.6,
    [switch]$PositiveOnly
) {
    $path = [System.Text.StringBuilder]::new()
    $started = $false
    foreach ($sample in $samples) {
        $xValue = [double]$sample.TimeMs
        $yValue = [double]$sample.$property
        if ([double]::IsNaN($yValue) -or [double]::IsInfinity($yValue) -or ($PositiveOnly -and $yValue -le 0.0)) {
            $started = $false
            continue
        }
        $px = Map-X $panel $xValue
        $py = Map-Y $panel $yValue
        if (-not $started) {
            [void]$path.Append('M' + (F $px) + ',' + (F $py))
            $started = $true
        } else {
            [void]$path.Append(' L' + (F $px) + ',' + (F $py))
        }
    }
    $dashAttribute = if ($dash) { ' stroke-dasharray="' + $dash + '"' } else { '' }
    [void]$builder.AppendLine('<path class="series" clip-path="url(#' + $panel.ClipId + ')" d="' + $path.ToString() + '" stroke="' + $color + '" stroke-width="' + (F $width) + '"' + $dashAttribute + '/>')
}

function Add-Threshold([System.Text.StringBuilder]$builder, $panel, [double]$value, [string]$label) {
    $py = Map-Y $panel $value
    [void]$builder.AppendLine('<line class="threshold" x1="' + (F $panel.PlotX) + '" y1="' + (F $py) + '" x2="' + (F ($panel.PlotX + $panel.PlotW)) + '" y2="' + (F $py) + '"/>')
    [void]$builder.AppendLine('<text class="annotation" x="' + (F ($panel.PlotX + $panel.PlotW - 5)) + '" y="' + (F ($py - 7)) + '" text-anchor="end" fill="#6B6B6B">' + (Escape-Xml $label) + '</text>')
}

function Add-Legend([System.Text.StringBuilder]$builder, [double]$x, [double]$y, [object[]]$items, [double]$spacing = 155.0) {
    $index = 0
    foreach ($item in $items) {
        $itemX = $x + $index * $spacing
        $dash = if ($item.Dash) { ' stroke-dasharray="' + $item.Dash + '"' } else { '' }
        [void]$builder.AppendLine('<line x1="' + (F $itemX) + '" y1="' + (F $y) + '" x2="' + (F ($itemX + 34)) + '" y2="' + (F $y) + '" stroke="' + $item.Color + '" stroke-width="2.8"' + $dash + '/>')
        [void]$builder.AppendLine('<text class="legend" x="' + (F ($itemX + 43)) + '" y="' + (F ($y + 5)) + '">' + (Escape-Xml $item.Label) + '</text>')
        $index++
    }
}

function Add-Annotation([System.Text.StringBuilder]$builder, [double]$x, [double]$y, [string]$text, [string]$anchor = 'start') {
    [void]$builder.AppendLine('<text class="annotation" x="' + (F $x) + '" y="' + (F $y) + '" text-anchor="' + $anchor + '">' + (Escape-Xml $text) + '</text>')
}

$water = Read-Diagnostics $WaterCsv 'Water' $waterColor
$sand = Read-Diagnostics $SandCsv 'Sand' $sandColor
$allSamples = @($water.Samples + $sand.Samples)
$xMax = [math]::Max(($water.Samples[-1].TimeMs), ($sand.Samples[-1].TimeMs))
$xTicks = @(0, 20, 40, 60, 80, 100) | Where-Object { $_ -le $xMax + 1.0e-9 }

# Figure 1: penetration depth
$maxDepth = ($allSamples | Measure-Object -Property PenetrationMm -Maximum).Maximum
$depthMax = [math]::Max(20.0, 20.0 * [math]::Ceiling($maxDepth / 20.0))
$depthTicks = for ($v = 0.0; $v -le $depthMax + 1.0e-9; $v += 20.0) { $v }
$svg = New-Svg 175 100 1400 800 'Hammer penetration depth for water and sand impact cases'
$panel = Add-Panel $svg 'penetration' 45 28 1310 730 0 $xMax 0 $depthMax $xTicks $depthTicks 'Time (ms)' 'Penetration depth (mm)' '(a)' 'Hammer penetration depth'
Add-Series $svg $panel $water.Samples 'PenetrationMm' $waterColor
Add-Series $svg $panel $sand.Samples 'PenetrationMm' $sandColor '11 7'
Add-Legend $svg ($panel.PlotX + 28) ($panel.PlotY + 30) @(
    [pscustomobject]@{Label='Water';Color=$waterColor;Dash=''},
    [pscustomobject]@{Label='Sand';Color=$sandColor;Dash='11 7'}
) 145
Close-Svg $svg (Join-Path $OutputDirectory 'fig01_penetration_depth.svg')

# Figure 2: action-reaction force and relative residual
$waterForceRange = Get-PropertyRange $water.Samples @('BodyForceZkN','OppositeMediumForceZkN')
$sandForceRange = Get-PropertyRange $sand.Samples @('BodyForceZkN','OppositeMediumForceZkN')
$waterForceScale = Get-NiceScale 0.0 $waterForceRange.Maximum 5 -ZeroBaseline
$sandForceScale = Get-NiceScale 0.0 $sandForceRange.Maximum 5 -ZeroBaseline
$svg = New-Svg 175 132 1400 1056 'Action-reaction force consistency for water and sand impact cases'
$pA = Add-Panel $svg 'ar-water-force' 20 20 680 505 0 $xMax $waterForceScale.Minimum $waterForceScale.Maximum $xTicks $waterForceScale.Ticks 'Time (ms)' 'Vertical force (kN)' '(a)' 'Water' 'linear'
$pB = Add-Panel $svg 'ar-sand-force' 700 20 680 505 0 $xMax $sandForceScale.Minimum $sandForceScale.Maximum $xTicks $sandForceScale.Ticks 'Time (ms)' 'Vertical force (kN)' '(b)' 'Sand' 'linear'
$logTicksAction = @(1e-10,1e-9,1e-8,1e-7,1e-6,1e-5)
$logFormatter = { param($v) '1e' + ([math]::Round([math]::Log10($v))).ToString($culture) }
$pC = Add-Panel $svg 'ar-water-residual' 20 525 680 505 0 $xMax 1e-10 1e-5 $xTicks $logTicksAction 'Time (ms)' 'Relative residual' '(c)' 'Water' 'log' { param($v) (F $v '0') } $logFormatter
$pD = Add-Panel $svg 'ar-sand-residual' 700 525 680 505 0 $xMax 1e-10 1e-5 $xTicks $logTicksAction 'Time (ms)' 'Relative residual' '(d)' 'Sand' 'log' { param($v) (F $v '0') } $logFormatter
Add-Series $svg $pA $water.Samples 'BodyForceZkN' $black
Add-Series $svg $pA $water.Samples 'OppositeMediumForceZkN' $waterColor '11 7'
Add-Series $svg $pB $sand.Samples 'BodyForceZkN' $black
Add-Series $svg $pB $sand.Samples 'OppositeMediumForceZkN' $sandColor '11 7'
Add-Series $svg $pC $water.ContactSamples 'ActionRelative' $waterColor '' 2.6 -PositiveOnly
Add-Series $svg $pD $sand.ContactSamples 'ActionRelative' $sandColor '' 2.6 -PositiveOnly
Add-Threshold $svg $pC 1e-5 'Limit'
Add-Threshold $svg $pD 1e-5 'Limit'
Add-Legend $svg ($pA.PlotX + 16) ($pA.PlotY + 26) @(
    [pscustomobject]@{Label='Body';Color=$black;Dash=''},
    [pscustomobject]@{Label='-Medium';Color=$waterColor;Dash='11 7'}
) 150
Add-Legend $svg ($pB.PlotX + 16) ($pB.PlotY + 26) @(
    [pscustomobject]@{Label='Body';Color=$black;Dash=''},
    [pscustomobject]@{Label='-Medium';Color=$sandColor;Dash='11 7'}
) 150
Close-Svg $svg (Join-Path $OutputDirectory 'fig02_action_reaction.svg')

# Figure 3: generalized-power residual
$waterAbsRange = Get-PropertyRange $water.ContactSamples @('GeneralizedAbsoluteW')
$sandAbsRange = Get-PropertyRange $sand.ContactSamples @('GeneralizedAbsoluteW')
$waterAbsScale = Get-NiceScale 0.0 $waterAbsRange.Maximum 5 -ZeroBaseline
$sandAbsScale = Get-NiceScale 0.0 $sandAbsRange.Maximum 5 -ZeroBaseline
$relTicks = @(1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1)
$svg = New-Svg 175 132 1400 1056 'Generalized contact power residual for water and sand impact cases'
$pA = Add-Panel $svg 'gp-water-abs' 20 20 680 505 0 $xMax $waterAbsScale.Minimum $waterAbsScale.Maximum $xTicks $waterAbsScale.Ticks 'Time (ms)' 'Absolute residual (W)' '(a)' 'Water' 'linear'
$pB = Add-Panel $svg 'gp-sand-abs' 700 20 680 505 0 $xMax $sandAbsScale.Minimum $sandAbsScale.Maximum $xTicks $sandAbsScale.Ticks 'Time (ms)' 'Absolute residual (W)' '(b)' 'Sand' 'linear'
$pC = Add-Panel $svg 'gp-water-rel' 20 525 680 505 0 $xMax 1e-7 1e1 $xTicks $relTicks 'Time (ms)' 'Relative residual' '(c)' 'Water' 'log' { param($v) (F $v '0') } $logFormatter
$pD = Add-Panel $svg 'gp-sand-rel' 700 525 680 505 0 $xMax 1e-7 1e1 $xTicks $relTicks 'Time (ms)' 'Relative residual' '(d)' 'Sand' 'log' { param($v) (F $v '0') } $logFormatter
Add-Series $svg $pA $water.ContactSamples 'GeneralizedAbsoluteW' $waterColor
Add-Series $svg $pB $sand.ContactSamples 'GeneralizedAbsoluteW' $sandColor
Add-Series $svg $pC $water.ContactSamples 'GeneralizedRelative' $waterColor '' 2.6 -PositiveOnly
Add-Series $svg $pD $sand.ContactSamples 'GeneralizedRelative' $sandColor '' 2.6 -PositiveOnly
Add-Threshold $svg $pC 1e-4 'Limit = 1e-4'
Add-Threshold $svg $pD 1e-4 'Limit = 1e-4'
Add-Annotation $svg ($pC.PlotX + 18) ($pC.PlotY + 27) ('Global L2 = ' + (F $water.GlobalPowerL2 '0.00E+0'))
Add-Annotation $svg ($pD.PlotX + 18) ($pD.PlotY + 27) ('Global L2 = ' + (F $sand.GlobalPowerL2 '0.00E+0'))
Close-Svg $svg (Join-Path $OutputDirectory 'fig03_generalized_power_residual.svg')

# Figure 4: energy balance
$waterEnergyRange = Get-PropertyRange $water.Samples @('DeltaMbdJ','MbdWorkJ','DeltaMediumJ','MediumWorkJ')
$sandEnergyRange = Get-PropertyRange $sand.Samples @('DeltaMbdJ','MbdWorkJ','DeltaMediumJ','MediumWorkJ')
$waterEnergyPadding = 0.05 * [math]::Max(1.0, $waterEnergyRange.Maximum - $waterEnergyRange.Minimum)
$sandEnergyPadding = 0.05 * [math]::Max(1.0, $sandEnergyRange.Maximum - $sandEnergyRange.Minimum)
$waterEnergyScale = Get-NiceScale ($waterEnergyRange.Minimum - $waterEnergyPadding) ($waterEnergyRange.Maximum + $waterEnergyPadding) 5
$sandEnergyScale = Get-NiceScale ($sandEnergyRange.Minimum - $sandEnergyPadding) ($sandEnergyRange.Maximum + $sandEnergyPadding) 5
$energyRelTicks = @(1e-4,1e-3,1e-2,1e-1,1e0)
$totalRelTicks = @(1e-3,1e-2,1e-1,1e0)
$svg = New-Svg 175 132 1400 1056 'Energy balance for water and sand impact cases'
$pA = Add-Panel $svg 'energy-water' 20 20 680 505 0 $xMax $waterEnergyScale.Minimum $waterEnergyScale.Maximum $xTicks $waterEnergyScale.Ticks 'Time (ms)' 'Energy or work (J)' '(a)' 'Water' 'linear'
$pB = Add-Panel $svg 'energy-sand' 700 20 680 505 0 $xMax $sandEnergyScale.Minimum $sandEnergyScale.Maximum $xTicks $sandEnergyScale.Ticks 'Time (ms)' 'Energy or work (J)' '(b)' 'Sand' 'linear'
$pC = Add-Panel $svg 'energy-mbd-residual' 20 525 680 505 0 $xMax 1e-4 1e0 $xTicks $energyRelTicks 'Time (ms)' 'MBD energy residual' '(c)' 'MBD balance' 'log' { param($v) (F $v '0') } $logFormatter
$pD = Add-Panel $svg 'energy-total-residual' 700 525 680 505 0 $xMax 1e-3 1e0 $xTicks $totalRelTicks 'Time (ms)' 'Total energy residual' '(d)' 'Coupled balance' 'log' { param($v) (F $v '0') } $logFormatter
foreach ($entry in @(@($pA,$water), @($pB,$sand))) {
    $p = $entry[0]
    $series = $entry[1]
    Add-Series $svg $p $series.Samples 'DeltaMbdJ' $quantityColors[0]
    Add-Series $svg $p $series.Samples 'MbdWorkJ' $quantityColors[1] '11 7'
    Add-Series $svg $p $series.Samples 'DeltaMediumJ' $quantityColors[2]
    Add-Series $svg $p $series.Samples 'MediumWorkJ' $quantityColors[3] '11 7'
}
Add-Series $svg $pC $water.Samples 'MbdEnergyRelative' $waterColor '' 2.6 -PositiveOnly
Add-Series $svg $pC $sand.Samples 'MbdEnergyRelative' $sandColor '11 7' 2.6 -PositiveOnly
Add-Series $svg $pD $water.Samples 'TotalEnergyRelative' $waterColor '' 2.6 -PositiveOnly
Add-Series $svg $pD $sand.Samples 'TotalEnergyRelative' $sandColor '11 7' 2.6 -PositiveOnly
Add-Threshold $svg $pC 2e-2 'Limit = 2%'
Add-Threshold $svg $pD 8e-2 'Limit = 8%'
$energyLegend = @(
    [pscustomobject]@{Label='dE_MBD';Color=$quantityColors[0];Dash=''},
    [pscustomobject]@{Label='W_MBD';Color=$quantityColors[1];Dash='11 7'},
    [pscustomobject]@{Label='dE_medium';Color=$quantityColors[2];Dash=''},
    [pscustomobject]@{Label='W_medium';Color=$quantityColors[3];Dash='11 7'}
)
Add-Legend $svg ($pA.PlotX + 10) ($pA.PlotY + 24) $energyLegend 123
Add-Legend $svg ($pB.PlotX + 10) ($pB.PlotY + 24) $energyLegend 123
Add-Legend $svg ($pC.PlotX + 22) ($pC.PlotY + 26) @(
    [pscustomobject]@{Label='Water';Color=$waterColor;Dash=''},
    [pscustomobject]@{Label='Sand';Color=$sandColor;Dash='11 7'}
) 145
Add-Legend $svg ($pD.PlotX + 22) ($pD.PlotY + 26) @(
    [pscustomobject]@{Label='Water';Color=$waterColor;Dash=''},
    [pscustomobject]@{Label='Sand';Color=$sandColor;Dash='11 7'}
) 145
Close-Svg $svg (Join-Path $OutputDirectory 'fig04_energy_balance.svg')

$plotColumns = @(
    'TimeMs','Contact','PenetrationMm','BodyForceZkN','OppositeMediumForceZkN',
    'ActionRelative','GeneralizedAbsoluteW','GeneralizedRelative','PowerScaleW',
    'DeltaMbdJ','MbdWorkJ','DeltaMediumJ','MediumWorkJ',
    'MbdEnergyRelative','TotalEnergyRelative'
)
$water.Samples | Select-Object -Property $plotColumns |
    Export-Csv -LiteralPath (Join-Path $OutputDirectory 'plot_data_water.csv') -NoTypeInformation -Encoding UTF8
$sand.Samples | Select-Object -Property $plotColumns |
    Export-Csv -LiteralPath (Join-Path $OutputDirectory 'plot_data_sand.csv') -NoTypeInformation -Encoding UTF8

$metrics = @(
    [pscustomobject]@{
        material = 'Water'
        maximum_penetration_mm = ($water.Samples | Measure-Object -Property PenetrationMm -Maximum).Maximum
        maximum_action_reaction_relative = ($water.ContactSamples | Measure-Object -Property ActionRelative -Maximum).Maximum
        generalized_power_global_l2 = $water.GlobalPowerL2
        final_mbd_energy_relative = $water.Samples[-1].MbdEnergyRelative
        final_total_energy_relative = $water.Samples[-1].TotalEnergyRelative
    },
    [pscustomobject]@{
        material = 'Sand'
        maximum_penetration_mm = ($sand.Samples | Measure-Object -Property PenetrationMm -Maximum).Maximum
        maximum_action_reaction_relative = ($sand.ContactSamples | Measure-Object -Property ActionRelative -Maximum).Maximum
        generalized_power_global_l2 = $sand.GlobalPowerL2
        final_mbd_energy_relative = $sand.Samples[-1].MbdEnergyRelative
        final_total_energy_relative = $sand.Samples[-1].TotalEnergyRelative
    }
)
$metrics | Export-Csv -LiteralPath (Join-Path $OutputDirectory 'figure_metrics.csv') -NoTypeInformation -Encoding UTF8

Write-Output "Generated publication figures in $OutputDirectory"
