param (
    [string]$Command = "help"
)

$AppDir = $PSScriptRoot
$AppFile = "app.py"

function Start-Server {
    Write-Host "Starting Sentify Server..." -ForegroundColor Green
    Set-Location -Path $AppDir
    
    # Start streamlit in a background job or separate process
    Start-Process -FilePath "streamlit" -ArgumentList "run $AppFile" -WindowStyle Hidden
    
    Write-Host "Server started in background! Waiting for it to initialize..." -ForegroundColor Green
    Start-Sleep -Seconds 3
    Write-Host "Opening browser..." -ForegroundColor Cyan
    Start-Process "http://localhost:8501"
}

function Stop-Server {
    Write-Host "Stopping the Sentify server..." -ForegroundColor Yellow
    
    # Find python processes running streamlit
    $streamlitProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' AND CommandLine LIKE '%streamlit%'"
    
    if ($streamlitProcesses) {
        foreach ($proc in $streamlitProcesses) {
            Write-Host "Terminating Streamlit process (PID: $($proc.ProcessId))..."
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Server stopped successfully." -ForegroundColor Green
    } else {
        Write-Host "No running Sentify server found." -ForegroundColor Yellow
    }
}

function Check-Status {
    $streamlitProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' AND CommandLine LIKE '%streamlit%'"
    
    if ($streamlitProcesses) {
        Write-Host "Status: " -NoNewline
        Write-Host "RUNNING" -ForegroundColor Green
        Write-Host "PIDs: $($streamlitProcesses.ProcessId -join ', ')"
        Write-Host "URL: http://localhost:8501"
    } else {
        Write-Host "Status: " -NoNewline
        Write-Host "STOPPED" -ForegroundColor Red
    }
}

switch ($Command.ToLower()) {
    "start" { Start-Server }
    "stop" { Stop-Server }
    "status" { Check-Status }
    "restart" {
        Stop-Server
        Start-Sleep -Seconds 2
        Start-Server
    }
    default {
        Write-Host "Sentify Management Tool (Windows PowerShell)" -ForegroundColor Cyan
        Write-Host "Usage: .\manage.ps1 [command]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  start    - Starts the Streamlit server in the background and opens browser"
        Write-Host "  stop     - Stops the running server"
        Write-Host "  status   - Checks if the server is running"
        Write-Host "  restart  - Stops then starts the server"
    }
}
