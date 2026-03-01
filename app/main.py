import asyncio
import threading
import uvicorn
import logging
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.logger import set_log_level, setup_logging
from app.config.settings import settings, APP_NAME, APP_VERSION, APP_DESCRIPTION
from app.middleware.logging import logging_middleware
from app.sessions.api import router as sessions_router
from app.api import router as chat_router
from app.llms.api import router as llms_router


# 创建FastAPI应用
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "deepLinking": True,
        "displayRequestDuration": True,
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
    }
)

# 确保日志配置在应用启动时被正确设置
setup_logging()

#==================================
# 注册所有路由器
#==================================
#app.include_router(llms_router, prefix="/api/v1", tags=["模型管理"])
app.include_router(sessions_router, prefix="/api/v1", tags=["Agent 会话管理"])
app.include_router(chat_router, prefix="/api/v1", tags=["聊天"])
app.include_router(llms_router, prefix="/api/v1", tags=["模型管理"])

#==================================
# 配置中间件
#==================================
# 配置CORS中间件 - 直接使用FastAPI内置的CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志中间件 - 直接使用全局中间件实例
app.add_middleware(logging_middleware)

#==================================
# 初始化基础设施
#==================================
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    try:
        logging.info("开始应用启动流程...")
        logging.info(f"{APP_NAME} v{APP_VERSION} 启动成功")

    except Exception as e:
        logging.error(f"应用启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    try:
        logging.info("应用正在关闭...")

    except Exception as e:
        logging.error(f"关闭连接失败: {e}")

# 根路径
@app.get("/")
async def root():
    """根路径 - 服务信息"""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
        "api_base": "/api/v1"
    }

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        # 基础服务状态检查
        health_status = {
            "status": "healthy",
            "service": APP_NAME,
            "version": APP_VERSION,
            "timestamp": datetime.now().isoformat(),
            "environment": "development" if settings.debug else "production"
        }

        return health_status
        
    except Exception as e:
        logging.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="服务不健康")

@app.post("/log-level")
async def change_log_level(level: str = Query(..., description="日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL")):
    """动态设置日志级别"""
    try:
        set_log_level(level)
        current_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())
        return {
            "message": f"日志级别已设置为 {level.upper()}",
            "current_level": current_level
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/log-level")
async def get_log_level():
    """获取当前日志级别"""
    current_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())
    return {
        "current_level": current_level
    }

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logging.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误"}
    )

def main():
    """主函数，用于启动服务器"""
    uvicorn.run(
        "app.main:app",
        host=settings.service_host,
        port=settings.service_port,
        reload=settings.debug
    )

if __name__ == "__main__":
    main() 