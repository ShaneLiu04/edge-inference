<#
.SYNOPSIS
  Run every test suite (calculator, harness, all labs) and print a pass/fail table.

.DESCRIPTION
  Each suite runs in its own pytest process so a missing dependency shows up as an ERROR
  row instead of aborting the whole run. C++ suites build with the g++ on PATH. Use
  -Python for a specific interpreter and -Only to filter suites by substring.

.EXAMPLE
  ./scripts/run-all-tests.ps1
  ./scripts/run-all-tests.ps1 -Only int8
#>
param(
    [string]$Python = "python",
    [string]$Only = ""
)

$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root;$($env:PYTHONPATH)" } else { $root }
$suites = [ordered]@{
    "tools edge_calc + harness"        = "tools/tests"
    "edge-runtime host-roofline"       = "solutions/edge-runtime/host-roofline/tests"
    "edge-runtime int8-kernels"        = "solutions/edge-runtime/int8-kernels/tests"
    "edge-runtime conv-kernels"        = "solutions/edge-runtime/conv-kernels/tests"
    "edge-runtime dma-double-buffer"   = "solutions/edge-runtime/dma-double-buffer/tests"
    "edge-runtime q4-gemv"             = "solutions/edge-runtime/q4-gemv/tests"
    "edge-runtime llama-cpp-7b"        = "solutions/edge-runtime/llama-cpp-7b/tests"
    "edge-runtime runtime-compare"     = "solutions/edge-runtime/runtime-compare/tests"
    "mit6.5940 lab2-quantization"      = "solutions/mit6.5940/lab2-quantization/tests"
    "mit6.5940 tinyengine-memory"      = "solutions/mit6.5940/tinyengine-memory/tests"
    "mlc tensor-program-tuning"        = "solutions/mlc/tensor-program-tuning/tests"
}

$rows = @()
foreach ($name in $suites.Keys) {
    if ($Only -and -not $name.Contains($Only)) { continue }
    $dir = Join-Path $root $suites[$name]
    if (-not (Test-Path $dir)) { $rows += [pscustomobject]@{Suite=$name; Result="MISSING"; Summary=""}; continue }
    Push-Location (Split-Path $dir)
    $out = & $Python -m pytest -q -p no:cacheprovider (Split-Path -Leaf $dir) 2>&1 | Out-String
    $code = $LASTEXITCODE
    Pop-Location
    $summary = ($out -split "`n" | Where-Object { $_ -match "passed|failed|error" } | Select-Object -Last 1)
    $summary = if ($summary) { $summary.Trim() } else { "" }
    $result = if ($code -eq 0) { "PASS" } elseif ($out -match "ModuleNotFoundError|ImportError") { "ERROR (deps)" } else { "FAIL" }
    if ($result -ne "PASS") { Write-Host ($out -split "`n" | Select-Object -Last 30 | Out-String) }
    $rows += [pscustomobject]@{Suite=$name; Result=$result; Summary=$summary}
}
$rows | Format-Table -AutoSize
if ($rows.Result -contains "FAIL" -or $rows.Result -contains "MISSING" -or $rows.Result -contains "ERROR (deps)") { exit 1 }
