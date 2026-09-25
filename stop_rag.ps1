# RAG 求职知识库助手 - 停止脚本
# 用法: .\stop_rag.ps1

Write-Host ""
Write-Host "停止 RAG 服务..." -ForegroundColor Yellow

# 停止 Milvus 容器（保留镜像和数据）
docker stop milvus-standalone 2>$null | Out-Null

Write-Host ""
Write-Host "提示：FastAPI 和 Streamlit 的窗口需手动关闭" -ForegroundColor Cyan
Write-Host "  - 找到标题带 'FastAPI 8001' 的窗口，按 Ctrl+C 或关闭窗口" -ForegroundColor Gray
Write-Host "  - 找到标题带 'Streamlit 8501' 的窗口，按 Ctrl+C 或关闭窗口" -ForegroundColor Gray
Write-Host ""
Write-Host "✅ Milvus 已停止" -ForegroundColor Green
Write-Host ""
Write-Host "  重新启动: .\start_rag.ps1" -ForegroundColor DarkGray
Write-Host ""