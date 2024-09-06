from typing import Optional, Callable
from ctypes import CDLL, Structure, cdll, c_char, c_size_t, c_int, c_bool
from pydantic import BaseModel, field_validator
from uuid import uuid4
from base58 import BITCOIN_ALPHABET

bitcoin_alphabet = BITCOIN_ALPHABET.decode("utf-8")

CharArray256 = c_char * 256
CharArray3 = c_char * 3
CharArray8 = c_char * 8


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
    enable_prefix: bool = False
    disable_prefix: bool = False
    prefix: str = ""
    suffix: str = ""
    max_attempts_per_kernel: int = 10000
    max_attempts_per_task: int = 1

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, prefix: str) -> str:
        if len(prefix) > 8:
            raise ValueError("Prefix must exceeds 8 characters.")
        if any(c not in bitcoin_alphabet and c != "?" for c in prefix):
            raise ValueError("Prefix must be in Base58 alphabet or ?.")
        if len(prefix) < 8:
            prefix += "?" * (8 - len(prefix))
        return prefix

    @field_validator("suffix")
    @classmethod
    def validate_suffix(cls, suffix: str) -> str:
        if len(suffix) > 8:
            raise ValueError("Suffix must exceeds 8 characters.")
        if any(c not in bitcoin_alphabet and c != "?" for c in suffix):
            raise ValueError("Suffix must be in Base58 alphabet or ?.")
        if len(suffix) < 8:
            suffix += "?" * (8 - len(suffix))
        return suffix

    @classmethod
    @field_validator("max_attempts_per_kernel")
    def validate_max_attempts_per_kernel(cls, max_attempts_per_kernel: int) -> int:
        if max_attempts_per_kernel <= 0:
            raise ValueError("Max attempts per kernel must be greater than 0.")
        return max_attempts_per_kernel

    @field_validator("max_attempts_per_task")
    @classmethod
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


class TaskStatus(BaseModel):
    finished: bool
    error: Optional[str] = None
    result: Optional[PyResult] = None


class CudaProcessor(object):

    dll: CDLL

    devices: Optional[int] = None
    device_lock: set[int] = set()

    info_cache: dict[int, PyDeviceInfo] = {}
    running_cache: set[str] = set()
    error_cache: dict[str, Exception] = {}
    result_cache: dict[str, PyResult] = {}

    def __init__(self, path: str):
        try:
            self.dll = cdll.LoadLibrary(path)
        except Exception as e:
            raise Exception(f"Failed to load library: {e}")

    def count_devices(self) -> int:
        if self.devices is not None:
            return self.devices

        func = self.dll.count_devices
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
            name=c_device_info.name.decode("utf-8"),
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
        task_id: str,
        task: PyTask,
    ):
        try:
            func = self.dll.vanity_run
            func.argtypes = [c_int, c_int, c_int, CTask]
            func.restype = CResult

            c_task = CTask(
                enable_prefix=task.enable_prefix,
                disable_prefix=task.disable_prefix,
                prefix=task.prefix.encode("utf-8"),
                suffix=task.suffix.encode("utf-8"),
                max_attempts_per_kernel=task.max_attempts_per_kernel,
                max_attempts_per_task=task.max_attempts_per_task,
            )
            c_result: CResult = func(c_int(device_id), c_int(blocks), c_int(threads_per_block), c_task)
            result = PyResult(
                found=c_result.found,
                pubkey=c_result.pubkey.decode("utf-8"),
                privkey=c_result.privkey.decode("utf-8"),
            )
            self.result_cache[task_id] = result
        except Exception as e:
            self.error_cache[task_id] = e
        finally:
            self.device_lock.discard(device_id)
            self.running_cache.discard(task_id)

    def assign_task(
        self,
        device_id: int,
        blocks: int,
        threads_per_block: int,
        task: PyTask,
        callback: Callable,
    ) -> str:
        if self.get_device_status(device_id):
            self.device_lock.add(device_id)
            task_id = str(uuid4())
            self.running_cache.add(task_id)
            callback(self.run_task, device_id, blocks, threads_per_block, task_id, task)
            return task_id
        else:
            raise Exception("Device is busy")

    def get_task_result(self, task_id: str) -> TaskStatus:
        if task_id in self.result_cache:
            result = self.result_cache[task_id]
            return TaskStatus(finished=True, result=result)
        elif task_id in self.running_cache:
            return TaskStatus(finished=False)
        elif task_id in self.error_cache:
            error = self.error_cache[task_id]
            return TaskStatus(finished=True, error=str(error))
        else:
            raise Exception("Task not found")

    def remove_task_output(self, task_id: str) -> None:
        if task_id in self.result_cache:
            del self.result_cache[task_id]
        elif task_id in self.error_cache:
            del self.error_cache[task_id]
        else:
            raise Exception("Result not found")


# if __name__ == "__main__":
#     processor = CudaProcessor("../lib/lib-pubkey-miner.so")
#     t = PyTask(
#         enable_prefix=True,
#         enable_suffix=True,
#         prefix="he",
#         max_attempts_per_task=100,
#     )
#     res = processor.run_task(0, 16, 256, t)
#     print(res.model_dump_json(indent=2))
