"""
This type stub file was generated by pyright.
"""

from typing import Dict, NamedTuple, Optional
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

class Config(NamedTuple):
    api_key: str = ...
    host: str = ...
    proxy_url: Optional[str] = ...
    proxy_headers: Optional[Dict[str, str]] = ...
    ssl_ca_certs: Optional[str] = ...
    ssl_verify: Optional[bool] = ...
    additional_headers: Optional[Dict[str, str]] = ...
    source_tag: Optional[str] = ...


class ConfigBuilder:
    """

    Configurations are resolved in the following order:

    - configs passed as keyword parameters
    - configs specified in environment variables
    - default values (if applicable)
    """
    @staticmethod
    def build(api_key: Optional[str] = ..., host: Optional[str] = ..., proxy_url: Optional[str] = ..., proxy_headers: Optional[Dict[str, str]] = ..., ssl_ca_certs: Optional[str] = ..., ssl_verify: Optional[bool] = ..., additional_headers: Optional[Dict[str, str]] = ..., **kwargs) -> Config:
        ...

    @staticmethod
    def build_openapi_config(config: Config, openapi_config: Optional[OpenApiConfiguration] = ..., **kwargs) -> OpenApiConfiguration:
        ...

