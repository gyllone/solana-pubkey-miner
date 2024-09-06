
from fastapi import FastAPI, applications
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from src.api import router as api_router


CDN_URL = "https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.9.0"


def swagger_monkey_patch(*args, **kwargs):
    """Swagger monkey patch"""
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url=f"{CDN_URL}/swagger-ui-bundle.js",
        swagger_css_url=f"{CDN_URL}/swagger-ui.css",
    )


applications.get_swagger_ui_html = swagger_monkey_patch


app = FastAPI(
    title="Solana Cuda Processor",
    description="A simple API for Solana Cuda Processor",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
