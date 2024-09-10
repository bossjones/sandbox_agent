"""
This type stub file was generated by pyright.
"""

class VCR:
    @staticmethod
    def is_test_method(method_name, function): # -> bool:
        ...

    @staticmethod
    def ensure_suffix(suffix): # -> Callable[..., Any]:
        ...

    def __init__(self, path_transformer=..., before_record_request=..., custom_patches=..., filter_query_parameters=..., ignore_hosts=..., record_mode=..., ignore_localhost=..., filter_headers=..., before_record_response=..., filter_post_data_parameters=..., match_on=..., before_record=..., inject_cassette=..., serializer=..., cassette_library_dir=..., func_path_generator=..., decode_compressed_response=..., record_on_exception=...) -> None:
        ...

    def use_cassette(self, path=..., **kwargs): # -> CassetteContextDecorator:
        ...

    def get_merged_config(self, **kwargs): # -> dict[str, Any]:
        ...

    def register_serializer(self, name, serializer): # -> None:
        ...

    def register_matcher(self, name, matcher): # -> None:
        ...

    def register_persister(self, persister): # -> None:
        ...

    def test_case(self, predicate=...): # -> type[temporary_class]:
        ...
