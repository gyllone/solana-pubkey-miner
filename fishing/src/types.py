from typing import Generic, TypeVar, Optional
from pydantic.generics import GenericModel


T = TypeVar("T")


class Response(GenericModel, Generic[T]):
    status: bool
    error: Optional[str] = None
    data: Optional[T] = None

    @classmethod
    def success(cls, data: T) -> "Response[T]":
        return cls[T](status=True, data=data)

    @classmethod
    def failure(cls, error: str) -> "Response[T]":
        return cls[T](status=False, error=error)
