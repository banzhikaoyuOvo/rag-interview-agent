# RAG 求职知识库助手 - Docker 停止
# 用法: .\stop_rag_docker.ps1

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host ""
Write-Host "停止 RAG 容器..." -ForegroundColor Yellow

# 停止 RAG 容器（保留 Milvus，因为 Milvus 被多个项目共用）
docker compose stop

Write-Host ""
Write-Host "✅ RAG 容器已停止（Milvus 保留运行）" -ForegroundColor Green
Write-Host ""
Write-Host "  重新启动: .\start_rag_docker.ps1" -ForegroundColor DarkGray
Write-Host "  彻底删除容器: docker compose down" -ForegroundColor DarkGray
Write-Host "  停止 Milvus: docker stop milvus-standalone" -ForegroundColor DarkGray
Write-Host ""