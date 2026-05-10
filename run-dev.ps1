# run-dev.ps1 - start worker and web proxy in separate PowerShell windows and open the UI
# Usage: from project root run: .\run-dev.ps1

# You may want to activate your conda env first (e.g. conda activate llm-gpu) before running this script.

$workerCmd = 'python Agent/worker.py'
$webCmd = 'uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload'

Write-Host "Starting worker..."
Start-Process -FilePath powershell -ArgumentList @('-NoExit','-Command',$workerCmd)

Start-Sleep -Milliseconds 500

Write-Host "Starting web proxy..."
Start-Process -FilePath powershell -ArgumentList @('-NoExit','-Command',$webCmd)

Start-Sleep -Seconds 1

Write-Host "Opening browser to http://127.0.0.1:8000"
Start-Process 'http://127.0.0.1:8000'
