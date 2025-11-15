# 设置控制台编码为 UTF-8
[System.Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[System.Console]::InputEncoding = [System.Text.Encoding]::UTF8

# 创建日志输出文件夹
$logDir = "docker_logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# 获取所有编号文件夹
$folders = Get-ChildItem -Directory | Where-Object { $_.Name -match "^\d+$" }

foreach ($folder in $folders) {
    $imageName = "docker_image_" + $folder.Name
    $containerName = "container_" + $folder.Name
    $logFile = Join-Path $logDir "$($folder.Name)_log.txt"

    Write-Host "Building Docker image: $imageName (path: $($folder.FullName))"
    docker build -t $imageName $folder.FullName

    Write-Host "Running Docker container: $containerName"
    docker run -d --name $containerName $imageName

    Start-Sleep -Seconds 3  # 可选：等待容器初始化

    Write-Host "Saving logs to: $logFile"
    docker logs $containerName > $logFile

    # 可选：如果容器不再需要，可以自动清理
    # docker rm -f $containerName
}

