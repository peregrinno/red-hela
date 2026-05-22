$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path "resources\references.json.gz")) {
    Write-Error "Coloque resources/references.json.gz antes de gerar os artefatos."
}

Write-Host "Pipeline offline: references + IVF (pode levar ~15 min)..."
docker run --rm `
    --memory=6g `
    --platform linux/amd64 `
    -v "${PWD}:/app" `
    -w /app `
    -e RED_HELA_ROOT=/app `
    -e PYTHONPATH=/app/src `
    -e PIP_IGNORE_INSTALLED=1 `
    python:3.12-slim `
    bash -lc "export DEBIAN_FRONTEND=noninteractive && apt-get update -qq && apt-get install -y -qq gcc g++ > /dev/null && pip install -q --no-cache-dir numpy ijson loguru scikit-learn && python -m red_hela.infrastructure.pack_references && python -m red_hela.infrastructure.pack_ivf"

Get-ChildItem resources\*.npy, resources\tree.npz | ForEach-Object {
    Write-Host ("  {0} ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB))
}
