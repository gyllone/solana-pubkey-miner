
from typing import Annotated, Optional
from fastapi import APIRouter, Body, Depends, BackgroundTasks

from src.processor import CudaProcessor, PyDeviceInfo, PyTask, TaskStatus
from src.types import Response
from src.settings import settings


router = APIRouter(
    prefix="/api/processor",
    tags=["Processor Endpoints"]
)


async def get_processor() -> CudaProcessor:
    return CudaProcessor(settings.LIP_PATH)


@router.get(
    "/count_devices",
    response_description="Get the number of GPU devices",
    response_model=Response[int],
    response_model_exclude_none=True,
)
async def count_devices(
    processor: Annotated[CudaProcessor, Depends(get_processor)]
) -> Response[int]:
    try:
        data = processor.count_devices()
        return Response.success(data)
    except Exception as e:
        return Response.failure(str(e))


@router.get(
    "/device_info",
    response_description="Get information about a specific device",
    response_model=Response[PyDeviceInfo],
    response_model_exclude_none=True,
)
async def get_device_info(
    processor: Annotated[CudaProcessor, Depends(get_processor)],
    device_id: int,
) -> Response[PyDeviceInfo]:
    try:
        data = processor.get_device_info(device_id)
        return Response.success(data)
    except Exception as e:
        return Response.failure(str(e))


@router.get(
    "/device_status",
    response_description="Check if a device is free now",
    response_model=Response[bool],
    response_model_exclude_none=True,
)
async def get_device_status(
    processor: Annotated[CudaProcessor, Depends(get_processor)],
    device_id: int,
) -> Response[bool]:
    try:
        data = processor.get_device_status(device_id)
        return Response.success(data)
    except Exception as e:
        return Response.failure(str(e))


@router.post(
    "/assign_task",
    response_description="Assign a task on a specific device",
    response_model=Response[str],
    response_model_exclude_none=True,
)
async def assign_task(
    background: BackgroundTasks,
    processor: Annotated[CudaProcessor, Depends(get_processor)],
    device_id: Annotated[int, Body()],
    blocks: Annotated[int, Body()],
    threads_per_block: Annotated[int, Body()],
    task: Annotated[PyTask, Body()],
) -> Response[str]:
    try:
        data = processor.assign_task(
            device_id,
            blocks,
            threads_per_block,
            task,
            background.add_task,
        )
        return Response.success(data)
    except Exception as e:
        return Response.failure(str(e))


@router.get(
    "/task_result",
    response_description="Get the result of a specific task",
    response_model=Response[TaskStatus],
    response_model_exclude_none=True,
)
async def get_task_result(
    processor: Annotated[CudaProcessor, Depends(get_processor)],
    task_id: str,
) -> Response[TaskStatus]:
    try:
        data = processor.get_task_result(task_id)
        return Response.success(data)
    except Exception as e:
        return Response.failure(str(e))


@router.delete(
    "/task_ouput",
    response_description="Remove a specific task result or error",
    response_model=Response[bool],
    response_model_exclude_none=True,
)
async def remove_task_output(
    processor: Annotated[CudaProcessor, Depends(get_processor)],
    task_id: str,
) -> Response[bool]:
    try:
        processor.remove_task_output(task_id)
        return Response.success(None)
    except Exception as e:
        return Response.failure(str(e))
