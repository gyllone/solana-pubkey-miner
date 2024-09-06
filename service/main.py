import click
import uvicorn

from src.app import app
from src.settings import settings


@click.command()
def main():
    """Run the server"""
    uvicorn.run(
        app,
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
    )


if __name__ == "__main__":
    main()
