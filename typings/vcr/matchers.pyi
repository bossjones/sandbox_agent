"""
This type stub file was generated by pyright.
"""

from typing import Set

_HEXDIG_CODE_POINTS: Set[int] = ...
log = ...
def method(r1, r2): # -> None:
    ...

def uri(r1, r2): # -> None:
    ...

def host(r1, r2): # -> None:
    ...

def scheme(r1, r2): # -> None:
    ...

def port(r1, r2): # -> None:
    ...

def path(r1, r2): # -> None:
    ...

def query(r1, r2): # -> None:
    ...

def raw_body(r1, r2): # -> None:
    ...

def body(r1, r2): # -> None:
    ...

def headers(r1, r2): # -> None:
    ...

_xml_header_checker = ...
_xmlrpc_header_checker = ...
_checker_transformer_pairs = ...
def requests_match(r1, r2, matchers): # -> bool:
    ...

def get_matchers_results(r1, r2, matchers): # -> tuple[list[Any], list[Any]]:
    """
    Get the comparison results of two requests as two list.
    The first returned list represents the matchers names that passed.
    The second list is the failed matchers as a string with failed assertion details if any.
    """
    ...

def get_assertion_message(assertion_details):
    """
    Get a detailed message about the failing matcher.
    """
    ...
