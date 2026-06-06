param(
    [int]$GpuId = 0,
    [int]$Port = 7861,
    [string]$ServerName = "127.0.0.1",
    [string]$EnvironmentName = "minigpt-crop",
    [switch]$NoBrowser,
    [switch]$Share
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Run the environment setup command from README.md first."
}

$requiredPaths = [ordered]@{
    "Vicuna 7B model" = "models\vicuna-7b\config.json"
    "crop disease checkpoint" = "models\checkpoint_0.pth"
    "EVA ViT-G checkpoint" = "models\eva_vit_g.pth"
    "English-to-Chinese translation model" = "models\opus-mt-en-zh\config.json"
}

$missing = @()
foreach ($entry in $requiredPaths.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value)) {
        $missing += "  - $($entry.Key): $($entry.Value)"
    }
}

if ($missing.Count -gt 0) {
    throw "Missing model files:`n$($missing -join "`n")`nSee models\README.md for the required layout."
}

$arguments = @(
    "run", "-n", $EnvironmentName, "--no-capture-output",
    "python", "-u", "demo_v2.py",
    "--cfg-path", "eval_configs/minigptv2_checkpoint_0_eval.yaml",
    "--gpu-id", $GpuId,
    "--server-name", $ServerName,
    "--server-port", $Port
)

if (-not $NoBrowser) {
    $arguments += "--inbrowser"
}
if ($Share) {
    $arguments += "--share"
}

Write-Host "Starting the crop diagnosis UI at http://${ServerName}:$Port/"
Write-Host "Model loading usually takes 1-3 minutes. Press Ctrl+C to stop."
& conda @arguments

if ($LASTEXITCODE -ne 0) {
    throw "The web service exited with code $LASTEXITCODE."
}
