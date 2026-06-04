param(
    [string]$ImageName = "trellis-classifier:latest",
    [string]$NodeContainer = "desktop-control-plane",
    [string]$DeploymentName = "trellis-classifier-api",
    [string]$ServiceName = "trellis-classifier-api",
    [int]$LocalPort = 8000,
    [switch]$PortForward,
    [switch]$SavePortForwardLogs,
    [string]$PortForwardLogDir = ".local/logs"
)

$ErrorActionPreference = "Stop"

# Runs a native command and raises when it exits non-zero.
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host $Description
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Description"
    }
}

# Starts kubectl port-forward in the foreground unless log capture is enabled.
function Start-LocalPortForward {
    $KubectlArgs = @("port-forward", "service/$ServiceName", "${LocalPort}:80")

    if (-not $SavePortForwardLogs) {
        Invoke-Native { kubectl @KubectlArgs } "Starting Kubernetes port-forward on localhost:$LocalPort"
        return
    }

    $LogDir = New-Item -ItemType Directory -Path $PortForwardLogDir -Force
    $OutLog = Join-Path $LogDir.FullName "port-forward.out.log"
    $ErrLog = Join-Path $LogDir.FullName "port-forward.err.log"

    $Process = Start-Process `
        -FilePath "kubectl" `
        -ArgumentList $KubectlArgs `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -PassThru `
        -WindowStyle Hidden

    Write-Host "Started background port-forward process: $($Process.Id)"
    Write-Host "stdout log: $OutLog"
    Write-Host "stderr log: $ErrLog"
}

Invoke-Native { docker build -t $ImageName . } "Building Docker image: $ImageName"

Invoke-Native {
    cmd.exe /d /s /c "docker save $ImageName | docker exec -i $NodeContainer ctr -n k8s.io images import -"
} "Importing image into Docker Desktop Kubernetes containerd"

Invoke-Native { kubectl rollout restart "deployment/$DeploymentName" } "Restarting Kubernetes deployment: $DeploymentName"
Invoke-Native { kubectl rollout status "deployment/$DeploymentName" } "Waiting for rollout to finish"

Write-Host "Current pods:"
kubectl get pods

Write-Host ""
if ($PortForward) {
    Start-LocalPortForward
} else {
    Write-Host "If pods are Running, expose the API with:"
    Write-Host "kubectl port-forward service/$ServiceName ${LocalPort}:80"
    Write-Host ""
    Write-Host "Or run this script with -PortForward. Add -SavePortForwardLogs only when file logs are needed."
}
