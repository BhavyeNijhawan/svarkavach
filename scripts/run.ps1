# SwarKavach one-command setup and launch (Windows PowerShell).
#
#   .\scripts\run.ps1              generate, train, evaluate, serve
#   .\scripts\run.ps1 -Serve       just start the console
#   .\scripts\run.ps1 -Fresh       wipe the corpus and models first
#   .\scripts\run.ps1 -N 800       bigger corpus

param(
    [switch]$Serve,
    [switch]$Fresh,
    [int]$N = 480,
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "no virtual environment found, creating one" -ForegroundColor Yellow
    # --system-site-packages reuses a torch already installed system wide,
    # which saves a 2.5 GB download on a machine that is short of disk
    python -m venv --system-site-packages "$root\.venv"
    & $py -m pip install --upgrade pip
    & $py -m pip install -r "$root\requirements.txt"
    & $py -m pip install -e "$root"
}

function Step($name, $block) {
    Write-Host ""
    Write-Host ("=" * 66) -ForegroundColor DarkGray
    Write-Host "  $name" -ForegroundColor Cyan
    Write-Host ("=" * 66) -ForegroundColor DarkGray
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$name failed with exit code $LASTEXITCODE" }
}

if ($Fresh) {
    Write-Host "removing the existing corpus, models and results" -ForegroundColor Yellow
    Remove-Item -Recurse -Force "$root\data\corpus\calls", "$root\data\corpus\audio",
                                "$root\data\models", "$root\data\results" -ErrorAction SilentlyContinue
}

if (-not $Serve) {
    $hasCorpus = Test-Path "$root\data\corpus\manifest.json"
    if ($Fresh -or -not $hasCorpus) {
        Step "Generating the corpus" { & $py -m swarkavach.cli gen-corpus --n $N --audio }
    } else {
        Write-Host "corpus already present, reusing it (pass -Fresh to rebuild)" -ForegroundColor DarkGray
    }

    $hasModels = Test-Path "$root\data\models\fusion_full_logreg.joblib"
    if ($Fresh -or -not $hasModels) {
        Step "Training" { & $py -m swarkavach.cli train }
    } else {
        Write-Host "models already present, reusing them (pass -Fresh to retrain)" -ForegroundColor DarkGray
    }

    Step "Evaluating" { & $py -m swarkavach.cli evaluate }
}

Write-Host ""
Write-Host "  Console: http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host ""
& $py -m swarkavach.cli serve --port $Port
