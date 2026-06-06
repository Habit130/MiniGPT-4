param(
    [string]$EnvironmentName = "minigpt-crop"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentFile = Join-Path $projectRoot "environment.yml"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Install Miniconda or Anaconda, then reopen PowerShell."
}

$environmentInfo = conda env list --json | ConvertFrom-Json
$environmentExists = $environmentInfo.envs | Where-Object {
    (Split-Path $_ -Leaf) -eq $EnvironmentName
}

if ($environmentExists) {
    Write-Host "Updating Conda environment $EnvironmentName ..."
    conda env update --name $EnvironmentName --file $environmentFile --prune
} else {
    Write-Host "Creating Conda environment $EnvironmentName ..."
    conda env create --name $EnvironmentName --file $environmentFile
}

if ($LASTEXITCODE -ne 0) {
    throw "Conda environment setup failed. Review the installation log above."
}

Write-Host "Validating core dependencies ..."
conda run -n $EnvironmentName python -c "import torch, torchvision, transformers, peft, bitsandbytes, gradio, cv2, timm, sentencepiece; print('Environment validation passed'); print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

if ($LASTEXITCODE -ne 0) {
    throw "The environment was created, but a core dependency could not be imported."
}

Write-Host "Environment is ready: $EnvironmentName"
