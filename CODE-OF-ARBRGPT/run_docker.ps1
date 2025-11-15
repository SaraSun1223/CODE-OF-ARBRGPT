# 设置代码页为UTF-8
[System.Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[System.Console]::InputEncoding = [System.Text.Encoding]::UTF8
# 获取所有编号文件夹
$folders = Get-ChildItem -Directory | Where-Object { $_.Name -match "^\d+$" }

foreach ($folder in $folders) {
    $imageName = "docker_image_" + $folder.Name
    $containerName = "container_" + $folder.Name

    Write-Host "builddocker: $imageName (path: $($folder.FullName))"
    docker build -t $imageName $folder.FullName

    Write-Host "rundocker: $containerName"
    docker run -d --name $containerName $imageName
    docker logs -f $containerName

}
