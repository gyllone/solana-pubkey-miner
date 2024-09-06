import asyncio
from typing import Optional
from ctypes import CDLL, Structure, cdll, c_char, c_size_t, c_int, c_bool
from pydantic import BaseModel, field_validator
from uuid import uuid4
from base58 import BITCOIN_ALPHABET
from asyncio import get_running_loop

CharArray256 = c_char * 256
CharArray3 = c_char * 3
CharArray8 = c_char * 8

bitcoin_alphabet = BITCOIN_ALPHABET.decode("utf-8")


class CDeviceInfo(Structure):
    _fields_ = [
        ("name", CharArray256),
        ("total_global_mem", c_size_t),
        ("max_blocks_per_multi_processor", c_int),
        ("max_threads_per_block", c_int),
        ("max_threads_dim", c_int),
        ("max_threads_per_multi_processor", c_int),
    ]


class PyDeviceInfo(BaseModel):
    name: str
    total_global_mem: int
    max_blocks_per_multi_processor: int
    max_threads_per_block: int
    max_threads_dim: int
    max_threads_per_multi_processor: int


class CTask(Structure):
    _fields_ = [
        ("enable_prefix", c_bool),
        ("disable_prefix", c_bool),
        ("prefix", CharArray8),
        ("suffix", CharArray8),
        ("max_attempts_per_kernel", c_int),
        ("max_attempts_per_task", c_int),
    ]


class PyTask(BaseModel):
    enable_prefix: bool
    disable_prefix: bool
    prefix: str
    suffix: str
    max_attempts_per_kernel: int
    max_attempts_per_task: int

    @classmethod
    @field_validator("prefix")
    def validate_prefix(cls, prefix: str) -> str:
        if len(prefix) != 8:
            raise ValueError("Prefix must be 8 characters long.")
        if any(c not in bitcoin_alphabet and c != "?" for c in prefix):
            raise ValueError("Prefix must be in Base58 alphabet or ?.")
        return prefix

    @classmethod
    @field_validator("suffix")
    def validate_suffix(cls, suffix: str) -> str:
        if len(suffix) != 8:
            raise ValueError("Suffix must be 8 characters long.")
        if any(c not in bitcoin_alphabet and c != "?" for c in suffix):
            raise ValueError("Suffix must be in Base58 alphabet or ?.")
        return suffix

    @classmethod
    @field_validator("max_attempts_per_kernel")
    def validate_max_attempts_per_kernel(cls, max_attempts_per_kernel: int) -> int:
        if max_attempts_per_kernel <= 0:
            raise ValueError("Max attempts per kernel must be greater than 0.")
        return max_attempts_per_kernel

    @classmethod
    @field_validator("max_attempts_per_task")
    def validate_max_attempts_per_task(cls, max_attempts_per_task: int) -> int:
        if max_attempts_per_task <= 0:
            raise ValueError("Max attempts per task must be greater than 0.")
        return max_attempts_per_task


class CResult(Structure):
    _fields_ = [
        ("found", c_bool),
        ("pubkey", CharArray256),
        ("privkey", CharArray256),
    ]


class PyResult(BaseModel):
    found:  bool
    pubkey: str
    privkey: str


class CudaProcessor(object):

    dll: CDLL

    devices: Optional[int] = None
    device_lock: set = set()

    info_cache: dict[int, PyDeviceInfo] = {}
    waiting_cache: dict[str, asyncio.Task] = {}
    result_cache: dict[str, PyResult] = {}

    def __init__(self, path: str):
        try:
            self.dll = cdll.LoadLibrary(path)
        except Exception as e:
            raise Exception(f"Failed to load library: {e}")

    def get_devices(self) -> int:
        if self.devices is not None:
            return self.devices

        func = self.dll.get_divices
        func.argtypes = []
        func.restype = c_int
        return func()

    def get_device_info(self, device_id: int) -> PyDeviceInfo:
        if device_id in self.info_cache:
            return self.info_cache[device_id]

        func = self.dll.get_device_info
        func.argtypes = [c_int]
        func.restype = CDeviceInfo

        c_device_info: CDeviceInfo = func(c_int(device_id))
        device_info = PyDeviceInfo(
            name=c_device_info.name.value.decode("utf-8"),
            total_global_mem=c_device_info.total_global_mem,
            max_blocks_per_multi_processor=c_device_info.max_blocks_per_multi_processor,
            max_threads_per_block=c_device_info.max_threads_per_block,
            max_threads_dim=c_device_info.max_threads_dim,
            max_threads_per_multi_processor=c_device_info.max_threads_per_multi_processor,
        )
        self.info_cache[device_id] = device_info
        return device_info

    def get_device_status(self, device_id: int) -> bool:
        return device_id not in self.device_lock

    def run_task(
        self,
        device_id: int,
        blocks: int,
        threads_per_block: int,
        task: PyTask,
    ) -> PyResult:
        if device_id in self.device_lock:
            raise Exception("Device is already in use.")
        else:
            self.device_lock.add(device_id)

        try:
            func = self.dll.vanity_run
            func.argtypes = [c_int, c_int, c_int, CTask]
            func.restype = CResult

            c_task = CTask(
                enable_prefix=task.enable_prefix,
                disable_prefix=task.disable_prefix,
                prefix=CharArray8(*task.prefix.encode("utf-8")),
                suffix=CharArray8(*task.suffix.encode("utf-8")),
                max_attempts_per_kernel=task.max_attempts_per_kernel,
                max_attempts_per_task=task.max_attempts_per_task,
            )
            c_result: CResult = func(c_int(device_id), c_int(blocks), c_int(threads_per_block), c_task)
            result = PyResult(
                found=c_result.found,
                pubkey=c_result.pubkey.value.decode("utf-8"),
                privkey=c_result.privkey.value.decode("utf-8"),
            )
            return result
        except Exception as e:
            raise e
        finally:
            self.device_lock.remove(device_id)

    async def async_run_task(
        self,
        device_id: int,
        blocks: int,
        threads_per_block: int,
        task: PyTask,
    ) -> PyResult:
        return await get_running_loop().run_in_executor(
            None,
            self.run_task,
            device_id,
            blocks,
            threads_per_block,
            task,
        )

    def assign_task(self, device_id: int, blocks: int, threads_per_block: int, task: PyTask) -> str:
        if self.get_device_status(device_id):
            _id = str(uuid4())
            waiting = asyncio.create_task(
                self.async_run_task(device_id, blocks, threads_per_block, task)
            )
            self.waiting_cache[_id] = waiting
            return _id
        else:
            raise Exception("Device is busy")

    def get_task_result(self, _id: str) -> Optional[PyResult]:
        waiting = self.waiting_cache.get(_id)
        if waiting is None:
            if _id not in self.result_cache:
                raise Exception("Task is not scheduled")
            return self.result_cache[_id]
        else:
            if not waiting.done():
                return None
            try:
                res = waiting.result()
                self.result_cache[_id] = res
                return res
            except Exception as e:
                raise Exception(f"Task run failed: {e}")
            finally:
                del self.waiting_cache[_id]

    def list_running_tasks(self) -> list[str]:
        return list(self.waiting_cache.keys())

    def list_task_results(self) -> list[str]:
        return list(self.result_cache.keys())

    def remove_task_result(self, _id: str) -> None:
        if _id in self.result_cache:
            del self.result_cache[_id]
        else:
            raise Exception("Result not found")
