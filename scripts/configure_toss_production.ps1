param(
    [switch]$Deploy
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

gh auth status | Out-Null

$clientKey = Read-Host "Toss Payments live client key"
$secretSecure = Read-Host "Toss Payments live secret key" -AsSecureString
$secretPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secretSecure)
try {
    $secretKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($secretPointer)
    if ($clientKey -notmatch '^live_ck_') {
        throw "Use the live API-individual client key (live_ck_), not a widget or test key."
    }
    if ($secretKey -notmatch '^live_sk_') {
        throw "Use the matching live API-individual secret key (live_sk_), not a widget or test key."
    }
    $clientKey | gh secret set TOSS_CLIENT_KEY --env production
    $secretKey | gh secret set TOSS_SECRET_KEY --env production
}
finally {
    if ($secretPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($secretPointer)
    }
    $secretKey = $null
    $secretSecure.Dispose()
}

Write-Host "Live Toss keys are stored in the protected production environment."
if ($Deploy) {
    gh workflow run release-readiness.yml --ref main
    Start-Sleep -Seconds 3
    $readiness = gh run list --workflow release-readiness.yml --branch main --limit 1 --json databaseId | ConvertFrom-Json
    if (-not $readiness.databaseId) { throw "Could not locate the release-readiness run." }
    gh run watch $readiness.databaseId --exit-status
    if ($LASTEXITCODE -ne 0) { throw "Release validation failed; production was not changed." }

    gh workflow run deploy-plan2bim-studio.yml --ref main
    Start-Sleep -Seconds 3
    $deployment = gh run list --workflow deploy-plan2bim-studio.yml --branch main --limit 1 --json databaseId | ConvertFrom-Json
    if (-not $deployment.databaseId) { throw "Could not locate the production deployment run." }
    gh run watch $deployment.databaseId --exit-status
    if ($LASTEXITCODE -ne 0) { throw "Production deployment failed." }
    Write-Host "Toss Payments is deployed. Complete one low-value live purchase and verify settlement in the merchant manager."
}
