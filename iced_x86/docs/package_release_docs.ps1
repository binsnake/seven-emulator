param(
    [string]$Version = "0.1.0",
    [string]$InputRoot = "build/docs/out",
    [string]$WarningsFile = "build/docs/doxygen-warnings.log",
    [string]$ReleaseRoot = "docs/release"
)

$ErrorActionPreference = "Stop"

$htmlSource = Join-Path $InputRoot "html"
$xmlSource = Join-Path $InputRoot "xml"

if (!(Test-Path $htmlSource)) {
    throw "Missing HTML docs: '$htmlSource'"
}
if (!(Test-Path $xmlSource)) {
    throw "Missing XML docs: '$xmlSource'"
}

$versionDir = Join-Path $ReleaseRoot $Version
$latestDir = Join-Path $ReleaseRoot "latest"

New-Item -ItemType Directory -Force $versionDir | Out-Null

$htmlOut = Join-Path $versionDir "html"
$xmlOut = Join-Path $versionDir "xml"

if (Test-Path $htmlOut) { Remove-Item -Recurse -Force $htmlOut }
if (Test-Path $xmlOut) { Remove-Item -Recurse -Force $xmlOut }

Copy-Item -Recurse -Force $htmlSource $htmlOut
Copy-Item -Recurse -Force $xmlSource $xmlOut

if (Test-Path $WarningsFile) {
    Copy-Item -Force $WarningsFile (Join-Path $versionDir "doxygen-warnings.log")
}

$manifest = @{
    project = "icedcpp"
    version = $Version
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
    paths = @{
        html = "html"
        xml = "xml"
        warnings = "doxygen-warnings.log"
    }
}
$manifestPath = Join-Path $versionDir "manifest.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content $manifestPath

$checksumPath = Join-Path $versionDir "checksums.sha256"
if (Test-Path $checksumPath) { Remove-Item -Force $checksumPath }
$files = Get-ChildItem -Recurse -File $versionDir | Sort-Object FullName
$versionRoot = (Get-Item $versionDir).FullName
$lines = foreach ($f in $files) {
    $hash = (Get-FileHash -Algorithm SHA256 $f.FullName).Hash.ToLowerInvariant()
    $full = $f.FullName
    $rel = $full.Substring($versionRoot.Length).TrimStart('\').Replace("\", "/")
    "$hash  $rel"
}
Set-Content $checksumPath $lines

if (Test-Path $latestDir) { Remove-Item -Recurse -Force $latestDir }
Copy-Item -Recurse -Force $versionDir $latestDir

Write-Host "Docs packaged:"
Write-Host "  Version: $versionDir"
Write-Host "  Latest : $latestDir"
