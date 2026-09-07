# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Fail if any PE inside a packaged Windows bundle is unsigned.
#
# This is the release gate. sign-windows-tree.ps1 signs the build output, but the
# packaging steps copy extra files in afterwards (the OpenMP runtime out of the
# Visual Studio redist tree, and the ROCm runtime DLLs out of the TheRock dist),
# so the only place that can prove what actually ships is the finished zip.
#
# Runs against zips rather than directories on purpose: an unsigned file that
# gets added between signing and packaging is invisible to any earlier check.

param(
    # Bundle zips to verify; every PE inside each is checked.
    [Parameter(Mandatory = $true)][string[]] $Path,
    # 7-Zip, preinstalled on the GitHub Windows images.
    [string] $SevenZip = '7z',
    # Known-unsigned leaf names to accept. Keep this empty. Anything listed here
    # is a file we ship that Smart App Control can still refuse to load.
    [string[]] $Allow = @()
)

$ErrorActionPreference = 'Continue'
$peExtensions = @('.exe', '.dll', '.pyd', '.sys', '.ocx', '.cpl', '.scr')

$unsigned = @()
$checked = 0

foreach ($bundle in $Path) {
    if (-not (Test-Path -LiteralPath $bundle -PathType Leaf)) {
        Write-Host "::error::bundle not found: $bundle"
        exit 1
    }
    $name = Split-Path $bundle -Leaf
    Write-Host ''
    Write-Host "=== $name ==="

    $root = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
    $dest = Join-Path $root ('sigcheck-' + [System.IO.Path]::GetFileNameWithoutExtension($name))
    Remove-Item -LiteralPath $dest -Recurse -Force -ErrorAction SilentlyContinue
    & $SevenZip x -y "-o$dest" $bundle | Out-Null
    # 7-Zip leaves a partial tree behind on error, so a created directory proves
    # nothing about whether the archive was fully extracted.
    if ($LASTEXITCODE -ne 0) {
        Write-Host "::error::7-Zip exited $LASTEXITCODE unpacking $name; contents not verified"
        exit 1
    }
    if (-not (Test-Path -LiteralPath $dest)) {
        Write-Host "::error::could not unpack $name; cannot verify its contents"
        exit 1
    }

    $inner = @(
        Get-ChildItem -LiteralPath $dest -Recurse -File |
            Where-Object { $peExtensions -contains $_.Extension.ToLowerInvariant() }
    )
    # No hits means the archive did not contain what we think it does, not that
    # the payload is clean.
    if ($inner.Count -eq 0) {
        Write-Host "::error::no executable payload found inside $name; contents not verified"
        exit 1
    }

    foreach ($f in ($inner | Sort-Object Name)) {
        $checked++
        $s = Get-AuthenticodeSignature -LiteralPath $f.FullName
        if ($s.Status -eq 'Valid') {
            Write-Host ('  signed    {0}' -f $f.Name)
        } elseif ($Allow -contains $f.Name) {
            Write-Host ('  ALLOWED   {0}  ({1}) - explicitly accepted as unsigned' -f $f.Name, $s.Status)
        } else {
            # UnknownError covers both "no signature at all" and "chain did not
            # build"; StatusMessage is what distinguishes them.
            Write-Host ('  UNSIGNED  {0}  ({1})  {2}' -f $f.Name, $s.Status, $s.StatusMessage)
            $unsigned += [pscustomobject]@{ Bundle = $name; File = $f.Name; Status = [string]$s.Status }
        }
    }
    Remove-Item -LiteralPath $dest -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host "checked $checked file(s) across $($Path.Count) bundle(s)"

if ($unsigned.Count -eq 0) {
    Write-Host 'Every PE in every bundle is validly signed.'
    exit 0
}

Write-Host ''
Write-Host '================ UNSIGNED FILES ================'
$unsigned | Format-Table Bundle, File, Status -AutoSize | Out-String | Write-Host
foreach ($u in $unsigned) {
    Write-Host "::error file=$($u.File)::$($u.File) in $($u.Bundle) is $($u.Status) and needs signing"
}
Write-Host ''
Write-Host 'These ship to user machines and are loaded by llama-server.'
Write-Host 'Smart App Control blocks unknown unsigned binaries as they load.'
exit 1
