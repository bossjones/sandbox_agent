"""
This type stub file was generated by pyright.
"""

from .util import CaseInsensitiveDict

"""
This type stub file was generated by pyright.
"""
log = ...
class Request:
    """
    VCR's representation of a request.
    """
    def __init__(self, method, uri, body, headers) -> None:
        ...

    @property
    def headers(self):
        ...

    @headers.setter
    def headers(self, value):
        ...

    @property
    def body(self):
        ...

    @body.setter
    def body(self, value):
        ...

    def add_header(self, key, value):
        ...

    @property
    def scheme(self):
        ...

    @property
    def host(self):
        ...

    @property
    def port(self):
        ...

    @property
    def path(self):
        ...

    @property
    def query(self):
        ...

    @property
    def url(self):
        ...

    @property
    def protocol(self):
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self):
        ...



class HeadersDict(CaseInsensitiveDict):
    """
    There is a weird quirk in HTTP.  You can send the same header twice.  For
    this reason, headers are represented by a dict, with lists as the values.
    However, it appears that HTTPlib is completely incapable of sending the
    same header twice.  This puts me in a weird position: I want to be able to
    accurately represent HTTP headers in cassettes, but I don't want the extra
    step of always having to do [0] in the general case, i.e.
    request.headers['key'][0]

    In addition, some servers sometimes send the same header more than once,
    and httplib *can* deal with this situation.

    Furthermore, I wanted to keep the request and response cassette format as
    similar as possible.

    For this reason, in cassettes I keep a dict with lists as keys, but once
    deserialized into VCR, I keep them as plain, naked dicts.
    """
    def __setitem__(self, key, value):
        ...