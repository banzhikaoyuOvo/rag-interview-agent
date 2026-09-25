# RAG 求职知识库助手 - 一键启动（Milvus + FastAPI + Streamlit）
# 用法: .\start_rag.ps1

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  🎯 RAG 求职知识库助手 - 启动" -ForegroundColor Cyan
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

# ── Step 2: 启动 Milvus ──
Write-Host ""
Write-Host "[2/4] 启动 Milvus..." -ForegroundColor Yellow
$milvusStatus = docker ps --filter "name=milvus-standalone" --format "{{.Status}}" 2>$null

if ($milvusStatus -match "healthy") {
    Write-Host "      ✅ Milvus 已在运行 (healthy)" -ForegroundColor Green
} else {
    if ($milvusStatus) {
        Write-Host "      当前状态: $milvusStatus，尝试重启..." -ForegroundColor Gray
        docker start milvus-standalone | Out-Null
    } else {
        Write-Host "      容器不存在，执行首次启动..." -ForegroundColor Gray
        & ".\standalone_embed.bat" start | Out-Null
    }

    Write-Host "      等待 healthy..." -ForegroundColor Gray
    $elapsed = 0
    $status = ""
    while ($elapsed -lt 60) {
        Start-Sleep -Seconds 3
        $elapsed += 3
        $status = docker ps --filter "name=milvus-standalone" --format "{{.Status}}" 2>$null
        if ($status -match "healthy") { break }
        if ($status -match "Exited|Restarting") {
            Write-Host "      ❌ Milvus 异常: $status" -ForegroundColor Red
            exit 1
        }
    }

    if ($status -match "healthy") {
        Write-Host "      ✅ Milvus 就绪 (耗时 ${elapsed}s)" -ForegroundColor Green
    } else {
        Write-Host "      ❌ Milvus 未就绪，超时 60s" -ForegroundColor Red
        Write-Host "      请检查: docker logs milvus-standalone --tail 30" -ForegroundColor Yellow
        exit 1
    }
}

# ── Step 3: 启动 FastAPI（新窗口）──
Write-Host ""
Write-Host "[3/4] 启动 FastAPI (端口 8001)..." -ForegroundColor Yellow
$fastapiCmd = "conda activate rag-interview; Set-Location '$projectRoot'; Write-Host '🎯 FastAPI 8001' -ForegroundColor Cyan; uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $fastapiCmd
Start-Sleep -Seconds 4
Write-Host "      ✅ FastAPI 已在新窗口启动" -ForegroundColor Green

# ── Step 4: 启动 Streamlit（新窗口）──
Write-Host ""
Write-Host "[4/4] 启动 Streamlit (端口 8501)..." -ForegroundColor Yellow
$streamlitCmd = "conda activate rag-interview; Set-Location '$projectRoot'; Write-Host '🎯 Streamlit 8501' -ForegroundColor Cyan; streamlit run src\web\streamlit_app.py --server.port 8501 --server.headless true"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $streamlitCmd
Start-Sleep -Seconds 5
Write-Host "      ✅ Streamlit 已在新窗口启动" -ForegroundColor Green

# ── Step 5: 打开浏览器 ──
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
Write-Host "  停止服务: .\stop_rag.ps1" -ForegroundColor DarkGray
Write-Host ""