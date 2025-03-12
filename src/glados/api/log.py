from litestar.middleware.logging import LoggingMiddlewareConfig
from litestar.plugins.structlog import StructlogConfig, StructlogPlugin

log_config = StructlogConfig(
    middleware_logging_config=LoggingMiddlewareConfig(
        response_log_fields=["status_code"],
    ),
)

structlog_plugin = StructlogPlugin(config=log_config)
