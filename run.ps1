param(
    [string]$ImageName = "lab2-house-segmentation",
    [int]$SyntheticSamples = 120,
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [switch]$SkipBuild,
    [switch]$SkipPrepare,
    [switch]$SkipTrain
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $projectRoot ".env"
$envExampleFile = Join-Path $projectRoot ".env.example"
$dataDir = Join-Path $projectRoot "data"
$artifactsDir = Join-Path $projectRoot "artifacts"
$modelPath = Join-Path $artifactsDir "best_model.pt"

Push-Location $projectRoot
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker is required but was not found on PATH."
    }

    if (-not (Test-Path $envFile)) {
        Copy-Item $envExampleFile $envFile
        Write-Host "Created .env from .env.example" -ForegroundColor Yellow
    }

    New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
    New-Item -ItemType Directory -Force -Path $artifactsDir | Out-Null

    if (-not $SkipBuild) {
        Invoke-CheckedCommand "Building Docker image" {
            docker build -t $ImageName .
        }
    }

    if (-not $SkipPrepare) {
        Invoke-CheckedCommand "Generating synthetic dataset" {
            docker run --rm `
                -v "${dataDir}:/app/data" `
                -v "${artifactsDir}:/app/artifacts" `
                $ImageName `
                python -m lab.prepare_dataset --generate-synthetic $SyntheticSamples
        }
    }

    if (-not $SkipTrain) {
        Invoke-CheckedCommand "Training model" {
            docker run --rm `
                -v "${dataDir}:/app/data" `
                -v "${artifactsDir}:/app/artifacts" `
                $ImageName `
                python -m lab.train --epochs $Epochs --batch-size $BatchSize
        }
    }

    if (-not (Test-Path $modelPath)) {
        throw "Model checkpoint not found at '$modelPath'. Run training first or remove -SkipTrain."
    }

    Write-Host "==> Starting API on http://localhost:8000" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop the container." -ForegroundColor DarkGray
    docker run --rm `
        -p 8000:8000 `
        --env-file $envFile `
        -v "${artifactsDir}:/app/artifacts" `
        $ImageName
}
finally {
    Pop-Location
}
