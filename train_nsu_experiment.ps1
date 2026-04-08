param(
    [Parameter(Mandatory = $true)]
    [string]$Name,

    [string]$Track = "glove_fusion",
    [string]$DataPath = "dataset_fusion_hold_v4_flags",
    [string]$SensorMode = "imu_flags",
    [string]$ModelVariant = "stable_hybrid",
    [string]$UseFlexPosture = "1",
    [string]$Seed = "42"
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyy-MM-dd"
$experimentName = "${timestamp}_${Name}"
$outputDir = Join-Path "experiments\$Track" $experimentName

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$env:NSU_DATA_PATH = $DataPath
$env:NSU_OUTPUT_DIR = $outputDir
$env:NSU_SENSOR_MODE = $SensorMode
$env:NSU_MODEL_VARIANT = $ModelVariant
$env:NSU_USE_FLEX_POSTURE = $UseFlexPosture
$env:NSU_RANDOM_SEED = $Seed

Write-Host "[Train] DataPath      : $($env:NSU_DATA_PATH)"
Write-Host "[Train] OutputDir     : $($env:NSU_OUTPUT_DIR)"
Write-Host "[Train] SensorMode    : $($env:NSU_SENSOR_MODE)"
Write-Host "[Train] ModelVariant  : $($env:NSU_MODEL_VARIANT)"
Write-Host "[Train] UseFlexPosture: $($env:NSU_USE_FLEX_POSTURE)"
Write-Host "[Train] Seed          : $($env:NSU_RANDOM_SEED)"

& "C:\SignProject\myenv\Scripts\python.exe" "C:\SignProject\nsu_train.py"

Write-Host ""
Write-Host "[Train] Finished: $outputDir"
