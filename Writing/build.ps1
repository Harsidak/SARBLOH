# build.ps1
Write-Host "Compiling LaTeX document..." -ForegroundColor Cyan
latexmk -pdf main.tex
Write-Host "Done! The output is in the output/ directory." -ForegroundColor Green
