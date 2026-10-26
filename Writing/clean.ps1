# clean.ps1
Write-Host "Cleaning up auxiliary files..." -ForegroundColor Cyan
latexmk -c
Write-Host "Cleaned!" -ForegroundColor Green
