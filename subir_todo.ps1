# Navegar a la carpeta del proyecto
Set-Location "C:\Users\ingos\OneDrive\Desktop\mask\mask"
Write-Host "Directorio actual: $(Get-Location)" -ForegroundColor Cyan

# Configurar Git para soportar subidas grandes y conexiones lentas
Write-Host "Configurando Git..." -ForegroundColor Cyan
git config http.postBuffer 524288000

# Deshacer el commit pesado anterior si existe
Write-Host "Limpiando preparación anterior..." -ForegroundColor Cyan
git reset --soft HEAD~1 2>$null
git restore --staged *.part* 2>$null

# Paso 1: Subir código base
Write-Host "Subiendo código base..." -ForegroundColor Yellow
git add .
git restore --staged *.part* 2>$null
git commit -m "Codigo base del detector"
git push -u origin main

# Paso 2: Subir Parte 3 (60MB)
Write-Host "Subiendo Parte 3 (60MB)..." -ForegroundColor Yellow
git add mask_detector_trained.h5.part3
git commit -m "Parte 3 del modelo"
git push

# Paso 3: Subir Parte 2 (80MB)
Write-Host "Subiendo Parte 2 (80MB)..." -ForegroundColor Yellow
git add mask_detector_trained.h5.part2
git commit -m "Parte 2 del modelo"
git push

# Paso 4: Subir Parte 1 (80MB)
Write-Host "Subiendo Parte 1 (80MB)..." -ForegroundColor Yellow
git add mask_detector_trained.h5.part1
git commit -m "Parte 1 del modelo"
git push

Write-Host "¡Subida a GitHub completada con éxito!" -ForegroundColor Green
