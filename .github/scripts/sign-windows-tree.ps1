# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Authenticode-sign every PE in a built Windows tree with Azure Trusted Signing.
#
# The bundles produced here are downloaded and executed on user machines by the
# Unsloth Studio installer, which fetches them after its own signed installer has
# already run. Nothing else signs them, so an unsigned file here reaches the user
# unsigned. Windows Smart App Control evaluates every binary as it loads and
# blocks unknown unsigned code, reporting it as a "Bad Image" dialog with status
# 0xc0e90002 naming whichever dependent DLL it refused, so signing only the
# launcher executables is not enough: every DLL in the tree has to be signed.
#
# Signing is batched. trusted-signing-cli takes any number of trailing paths and
# authenticates to Azure once per invocation, so a bundle of ~50 files costs one
# round trip instead of fifty.

param(
    # Directory to sign, searched recursively.
    [Parameter(Mandatory = $true)][string] $Path,
    # Azure Trusted Signing endpoint, matched to the account's region.
    [string] $Endpoint = 'https://eus.codesigning.azure.net',
    # Signature description shown in the Windows UAC/properties dialog.
    [string] $Description = 'Unsloth',
    # Files per trusted-signing-cli invocation. Batching amortizes Azure auth;
    # an unbounded batch would risk the Windows command line length limit.
    [int] $BatchSize = 40,
    [int] $MaxAttempts = 3
)

$ErrorActionPreference = 'Continue'

# Extensions Smart App Control and WDAC evaluate at load time. .pyd is included
# because the ROCm and CUDA bundles can carry Python extension modules, which are
# ordinary PEs under a different suffix.
$peExtensions = @('.exe', '.dll', '.pyd', '.sys', '.ocx', '.cpl', '.scr')

if (-not (Test-Path -LiteralPath $Path)) {
    Write-Host "::error::sign-windows-tree: path not found: $Path"
    exit 1
}

$files = @(
    Get-ChildItem -LiteralPath $Path -Recurse -File |
        Where-Object { $peExtensions -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object FullName
)

# An empty tree means the build step silently produced nothing. Signing zero
# files and reporting success would let that reach the release.
if ($files.Count -eq 0) {
    Write-Host "::error::sign-windows-tree: no PE files found under $Path; nothing was built"
    exit 1
}

Write-Host "signing $($files.Count) PE file(s) under $Path"

# Retried only for Azure auth flakiness. A signing rejection is a real failure
# and repeating it just burns quota.
$retryPatterns = @(
    'No subscriptions found',
    'login via azure cli',
    'az\.cmd.*exited with code 1',
    'Failed to acquire token',
    'temporarily unavailable',
    'Response status code does not indicate success: 429',
    'Response status code does not indicate success: 50[0-9]'
)

$batches = [System.Collections.Generic.List[object]]::new()
for ($i = 0; $i -lt $files.Count; $i += $BatchSize) {
    $end = [Math]::Min($i + $BatchSize, $files.Count) - 1
    $batches.Add(@($files[$i..$end]))
}

$batchNumber = 0
foreach ($batch in $batches) {
    $batchNumber++
    $paths = @($batch | ForEach-Object { $_.FullName })
    Write-Host ''
    Write-Host "=== batch $batchNumber/$($batches.Count): $($paths.Count) file(s) ==="

    $signed = $false
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        $cliArgs = @('-e', $Endpoint, '-d', $Description) + $paths
        $output = & trusted-signing-cli @cliArgs 2>&1
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) { $exitCode = 1 }
        $text = $output | Out-String
        foreach ($line in $output) { Write-Output $line }

        if ($exitCode -eq 0) { $signed = $true; break }

        $isRetryable = $false
        foreach ($pattern in $retryPatterns) {
            if ($text -match $pattern) { $isRetryable = $true; break }
        }
        if (-not $isRetryable -or $attempt -eq $MaxAttempts) {
            Write-Host "::error::trusted-signing-cli exited $exitCode on batch $batchNumber"
            exit $exitCode
        }
        Write-Warning "trusted-signing-cli hit a transient Azure error; retrying batch $batchNumber."
        Start-Sleep -Seconds (5 * $attempt)
    }

    if (-not $signed) {
        Write-Host "::error::batch $batchNumber was not signed"
        exit 1
    }
}

Write-Host ''
Write-Host "signed $($files.Count) file(s); verifying"

# Verify here as well as in the release gate. Catching an unsigned file in the
# job that produced it names the build leg directly, where the gate downstream
# can only say which bundle was wrong.
$bad = @()
foreach ($f in $files) {
    $sig = Get-AuthenticodeSignature -LiteralPath $f.FullName
    if ($sig.Status -ne 'Valid') {
        $bad += [pscustomobject]@{ File = $f.Name; Status = [string]$sig.Status; Message = $sig.StatusMessage }
    }
}

if ($bad.Count -gt 0) {
    Write-Host ''
    $bad | Format-Table File, Status, Message -AutoSize | Out-String | Write-Host
    foreach ($b in $bad) {
        Write-Host "::error file=$($b.File)::$($b.File) is $($b.Status) after signing"
    }
    exit 1
}

Write-Host "all $($files.Count) file(s) report a valid Authenticode signature"
exit 0
