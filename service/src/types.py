from typing import Generic, TypeVar, Optional
from pydantic import BaseModel


T = TypeVar("T")


class Response(BaseModel, Generic[T]):
    status: bool
    error: Optional[str] = None
    data: Optional[T] = None

    @classmethod
    def success(cls, data: T) -> "Response[T]":
        return cls[T](status=True, data=data)

    @classmethod
    def failure(cls, error: str) -> "Response[T]":
        return cls[T](status=False, error=error)
