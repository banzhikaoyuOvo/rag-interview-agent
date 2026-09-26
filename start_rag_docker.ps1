# RAG 求职知识库助手 - Docker 启动（Milvus + rag-api + rag-web）
# 用法: .\start_rag_docker.ps1

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  🐳 RAG 求职知识库助手 - Docker 启动" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# ── Step 1: 检查 Docker ──
Write-Host ""
Write-Host "[1/4] 检查 Docker Desktop..." -ForegroundColor Yellow
try {
    docker info *>$null
    if ($LASTEXITCODE -ne 0) { throw "Docker 未运行" }
    Write-Host "      ✅ Docker 正在运行" -ForegroundColor Green
} catch {
    Write-Host "      ❌ Docker Desktop 未启动" -ForegroundColor Red
    Write-Host "      请先打开 Docker Desktop，等鲸鱼图标变绿后重试" -ForegroundColor Yellow
    exit 1
}

# ── Step 2: 检查 Milvus ──
Write-Host ""
Write-Host "[2/4] 检查 Milvus..." -ForegroundColor Yellow
$milvusStatus = docker ps --filter "name=milvus-standalone" --format "{{.Status}}" 2>$null

if ($milvusStatus -match "healthy") {
    Write-Host "      ✅ Milvus 已在运行 (healthy)" -ForegroundColor Green
} else {
    if ($milvusStatus) {
        docker start milvus-standalone | Out-Null
    } else {
        & ".\standalone_embed.bat" start | Out-Null
    }
    Write-Host "      等待 healthy..." -ForegroundColor Gray
    $elapsed = 0
    while ($elapsed -lt 60) {
        Start-Sleep -Seconds 3
        $elapsed += 3
        $status = docker ps --filter "name=milvus-standalone" --format "{{.Status}}" 2>$null
        if ($status -match "healthy") { break }
    }
    if ($status -match "healthy") {
        Write-Host "      ✅ Milvus 就绪 (耗时 ${elapsed}s)" -ForegroundColor Green
    } else {
        Write-Host "      ❌ Milvus 未就绪" -ForegroundColor Red
        exit 1
    }
}

# ── Step 3: 启动 RAG 容器 ──
Write-Host ""
Write-Host "[3/4] 启动 RAG 容器 (rag-api + rag-web)..." -ForegroundColor Yellow
docker compose up -d

Write-Host "      等待 rag-api healthy..." -ForegroundColor Gray
$elapsed = 0
$apiHealthy = $false
while ($elapsed -lt 60) {
    Start-Sleep -Seconds 3
    $elapsed += 3
    $apiStatus = docker ps --filter "name=rag-api" --format "{{.Status}}" 2>$null
    if ($apiStatus -match "healthy") {
        $apiHealthy = $true
        Write-Host "      ✅ rag-api 就绪 (耗时 ${elapsed}s)" -ForegroundColor Green
        break
    }
    if ($apiStatus -match "Exited|Restarting") {
        Write-Host "      ❌ rag-api 异常: $apiStatus" -ForegroundColor Red
        Write-Host "      查看日志: docker logs rag-api --tail 30" -ForegroundColor Yellow
        exit 1
    }
}

if (-not $apiHealthy) {
    Write-Host "      ⚠️  rag-api 未在 60s 内 healthy，继续检查..." -ForegroundColor Yellow
}

# ── Step 4: 显示状态 + 打开浏览器 ──
Write-Host ""
Write-Host "[4/4] 服务状态:" -ForegroundColor Yellow
docker compose ps

Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ✅ 启动完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  🎯 Streamlit:  http://localhost:8501" -ForegroundColor White
Write-Host "  📖 Swagger UI: http://localhost:8001/docs" -ForegroundColor White
Write-Host "  💚 健康检查:   http://localhost:8001/api/health" -ForegroundColor White
Write-Host ""
Write-Host "  查看日志: docker logs rag-api -f" -ForegroundColor DarkGray
Write-Host "  停止服务: .\stop_rag_docker.ps1" -ForegroundColor DarkGray
Write-Host ""