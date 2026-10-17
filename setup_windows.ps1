param([string]$PythonBin = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($PythonBin) { & $PythonBin -m venv .venv } else { py -3.12 -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 environment creation failed.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; do not change pinned versions silently.' }
& .\.venv\Scripts\python.exe -B reproduce.py prepare
if ($LASTEXITCODE -ne 0) { throw 'Input checks failed.' }
& .\.venv\Scripts\python.exe -B reproduce.py preflight
if ($LASTEXITCODE -ne 0) { throw 'Dependency preflight failed.' }
