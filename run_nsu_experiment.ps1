param(
    [Parameter(Mandatory = $true)]
    [string]$ExperimentDir
)

$ErrorActionPreference = "Stop"

$resolvedExperimentDir = Resolve-Path $ExperimentDir
$experimentName = Split-Path $resolvedExperimentDir -Leaf
$debugDir = Join-Path "experiments\live_eval\$experimentName" "run_debug_captures"

New-Item -ItemType Directory -Force -Path $debugDir | Out-Null

$env:NSU_RUN_MODEL_DIR = $resolvedExperimentDir
$env:NSU_RUN_DEBUG_CAPTURE_DIR = $debugDir

Write-Host "[Run] ModelDir : $($env:NSU_RUN_MODEL_DIR)"
Write-Host "[Run] DebugDir : $($env:NSU_RUN_DEBUG_CAPTURE_DIR)"

& "C:\SignProject\myenv\Scripts\python.exe" "C:\SignProject\nsu_run_fusion_v6_overlap_hold.py"
