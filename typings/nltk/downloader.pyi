"""
This type stub file was generated by pyright.
"""

import threading

"""
The NLTK corpus and module downloader.  This module defines several
interfaces which can be used to download corpora, models, and other
data packages that can be used with NLTK.

Downloading Packages
====================
If called with no arguments, ``download()`` will display an interactive
interface which can be used to download and install new packages.
If Tkinter is available, then a graphical interface will be shown,
otherwise a simple text interface will be provided.

Individual packages can be downloaded by calling the ``download()``
function with a single argument, giving the package identifier for the
package that should be downloaded:

    >>> download('treebank') # doctest: +SKIP
    [nltk_data] Downloading package 'treebank'...
    [nltk_data]   Unzipping corpora/treebank.zip.

NLTK also provides a number of \"package collections\", consisting of
a group of related packages.  To download all packages in a
colleciton, simply call ``download()`` with the collection's
identifier:

    >>> download('all-corpora') # doctest: +SKIP
    [nltk_data] Downloading package 'abc'...
    [nltk_data]   Unzipping corpora/abc.zip.
    [nltk_data] Downloading package 'alpino'...
    [nltk_data]   Unzipping corpora/alpino.zip.
      ...
    [nltk_data] Downloading package 'words'...
    [nltk_data]   Unzipping corpora/words.zip.

Download Directory
==================
By default, packages are installed in either a system-wide directory
(if Python has sufficient access to write to it); or in the current
user's home directory.  However, the ``download_dir`` argument may be
used to specify a different installation target, if desired.

See ``Downloader.default_download_dir()`` for more a detailed
description of how the default download directory is chosen.

NLTK Download Server
====================
Before downloading any packages, the corpus and module downloader
contacts the NLTK download server, to retrieve an index file
describing the available packages.  By default, this index file is
loaded from ``https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml``.
If necessary, it is possible to create a new ``Downloader`` object,
specifying a different URL for the package index file.

Usage::

    python nltk/downloader.py [-d DATADIR] [-q] [-f] [-k] PACKAGE_IDS

or::

    python -m nltk.downloader [-d DATADIR] [-q] [-f] [-k] PACKAGE_IDS
"""
TKINTER = ...
class Package:
    """
    A directory entry for a downloadable package.  These entries are
    extracted from the XML index file that is downloaded by
    ``Downloader``.  Each package consists of a single file; but if
    that file is a zip file, then it can be automatically decompressed
    when the package is installed.
    """
    def __init__(self, id, url, name=..., subdir=..., size=..., unzipped_size=..., checksum=..., svn_revision=..., copyright=..., contact=..., license=..., author=..., unzip=..., **kw) -> None:
        ...

    @staticmethod
    def fromxml(xml): # -> Package:
        ...

    def __lt__(self, other) -> bool:
        ...

    def __repr__(self): # -> LiteralString:
        ...



class Collection:
    """
    A directory entry for a collection of downloadable packages.
    These entries are extracted from the XML index file that is
    downloaded by ``Downloader``.
    """
    def __init__(self, id, children, name=..., **kw) -> None:
        ...

    @staticmethod
    def fromxml(xml): # -> Collection:
        ...

    def __lt__(self, other) -> bool:
        ...

    def __repr__(self): # -> LiteralString:
        ...



class DownloaderMessage:
    """A status message object, used by ``incr_download`` to
    communicate its progress."""
    ...


class StartCollectionMessage(DownloaderMessage):
    """Data server has started working on a collection of packages."""
    def __init__(self, collection) -> None:
        ...



class FinishCollectionMessage(DownloaderMessage):
    """Data server has finished working on a collection of packages."""
    def __init__(self, collection) -> None:
        ...



class StartPackageMessage(DownloaderMessage):
    """Data server has started working on a package."""
    def __init__(self, package) -> None:
        ...



class FinishPackageMessage(DownloaderMessage):
    """Data server has finished working on a package."""
    def __init__(self, package) -> None:
        ...



class StartDownloadMessage(DownloaderMessage):
    """Data server has started downloading a package."""
    def __init__(self, package) -> None:
        ...



class FinishDownloadMessage(DownloaderMessage):
    """Data server has finished downloading a package."""
    def __init__(self, package) -> None:
        ...



class StartUnzipMessage(DownloaderMessage):
    """Data server has started unzipping a package."""
    def __init__(self, package) -> None:
        ...



class FinishUnzipMessage(DownloaderMessage):
    """Data server has finished unzipping a package."""
    def __init__(self, package) -> None:
        ...



class UpToDateMessage(DownloaderMessage):
    """The package download file is already up-to-date"""
    def __init__(self, package) -> None:
        ...



class StaleMessage(DownloaderMessage):
    """The package download file is out-of-date or corrupt"""
    def __init__(self, package) -> None:
        ...



class ErrorMessage(DownloaderMessage):
    """Data server encountered an error"""
    def __init__(self, package, message) -> None:
        ...



class ProgressMessage(DownloaderMessage):
    """Indicates how much progress the data server has made"""
    def __init__(self, progress) -> None:
        ...



class SelectDownloadDirMessage(DownloaderMessage):
    """Indicates what download directory the data server is using"""
    def __init__(self, download_dir) -> None:
        ...



class Downloader:
    """
    A class used to access the NLTK data server, which can be used to
    download corpora and other data packages.
    """
    INDEX_TIMEOUT = ...
    DEFAULT_URL = ...
    INSTALLED = ...
    NOT_INSTALLED = ...
    STALE = ...
    PARTIAL = ...
    def __init__(self, server_index_url=..., download_dir=...) -> None:
        ...

    def list(self, download_dir=..., show_packages=..., show_collections=..., header=..., more_prompt=..., skip_installed=...): # -> None:
        ...

    def packages(self): # -> dict_values[Any, Package]:
        ...

    def corpora(self): # -> list[Package]:
        ...

    def models(self): # -> list[Package]:
        ...

    def collections(self): # -> dict_values[Any, Collection]:
        ...

    def incr_download(self, info_or_id, download_dir=..., force=...): # -> Generator[SelectDownloadDirMessage | ErrorMessage | ProgressMessage | Any | StartCollectionMessage | FinishCollectionMessage | StartPackageMessage | UpToDateMessage | FinishPackageMessage | StaleMessage | StartDownloadMessage | FinishDownloadMessage | StartUnzipMessage | FinishUnzipMessage, Any, None]:
        ...

    def download(self, info_or_id=..., download_dir=..., quiet=..., force=..., prefix=..., halt_on_error=..., raise_on_error=..., print_error_to=...): # -> bool:
        ...

    def is_stale(self, info_or_id, download_dir=...): # -> bool:
        ...

    def is_installed(self, info_or_id, download_dir=...): # -> bool:
        ...

    def clear_status_cache(self, id=...): # -> None:
        ...

    def status(self, info_or_id, download_dir=...): # -> Literal['out of date', 'partial', 'not installed', 'installed']:
        """
        Return a constant describing the status of the given package
        or collection.  Status can be one of ``INSTALLED``,
        ``NOT_INSTALLED``, ``STALE``, or ``PARTIAL``.
        """
        ...

    def update(self, quiet=..., prefix=...): # -> None:
        """
        Re-download any packages whose status is STALE.
        """
        ...

    def index(self): # -> None:
        """
        Return the XML index describing the packages available from
        the data server.  If necessary, this index will be downloaded
        from the data server.
        """
        ...

    def info(self, id): # -> Package | Collection:
        """Return the ``Package`` or ``Collection`` record for the
        given item."""
        ...

    def xmlinfo(self, id):
        """Return the XML info record for the given item"""
        ...

    url = ...
    def default_download_dir(self): # -> str | None:
        """
        Return the directory to which packages will be downloaded by
        default.  This value can be overridden using the constructor,
        or on a case-by-case basis using the ``download_dir`` argument when
        calling ``download()``.

        On Windows, the default download directory is
        ``PYTHONHOME/lib/nltk``, where *PYTHONHOME* is the
        directory containing Python, e.g. ``C:\\Python25``.

        On all other platforms, the default directory is the first of
        the following which exists or which can be created with write
        permission: ``/usr/share/nltk_data``, ``/usr/local/share/nltk_data``,
        ``/usr/lib/nltk_data``, ``/usr/local/lib/nltk_data``, ``~/nltk_data``.
        """
        ...

    download_dir = ...


class DownloaderShell:
    def __init__(self, dataserver) -> None:
        ...

    def run(self): # -> None:
        ...



class DownloaderGUI:
    """
    Graphical interface for downloading packages from the NLTK data
    server.
    """
    COLUMNS = ...
    COLUMN_WEIGHTS = ...
    COLUMN_WIDTHS = ...
    DEFAULT_COLUMN_WIDTH = ...
    INITIAL_COLUMNS = ...
    _BACKDROP_COLOR = ...
    _ROW_COLOR = ...
    _MARK_COLOR = ...
    _FRONT_TAB_COLOR = ...
    _BACK_TAB_COLOR = ...
    _PROGRESS_COLOR = ...
    _TAB_FONT = ...
    def __init__(self, dataserver, use_threads=...) -> None:
        ...

    _tab = ...
    _rows = ...
    _DL_DELAY = ...
    def destroy(self, *e): # -> None:
        ...

    def mainloop(self, *args, **kwargs): # -> None:
        ...

    HELP = ...
    def help(self, *e): # -> None:
        ...

    def about(self, *e): # -> None:
        ...

    _gradient_width = ...
    class _DownloadThread(threading.Thread):
        def __init__(self, data_server, items, lock, message_queue, abort) -> None:
            ...

        def run(self): # -> None:
            ...



    _MONITOR_QUEUE_DELAY = ...


def md5_hexdigest(file): # -> str:
    """
    Calculate and return the MD5 checksum for a given file.
    ``file`` may either be a filename or an open stream.
    """
    ...

def unzip(filename, root, verbose=...): # -> None:
    """
    Extract the contents of the zip file ``filename`` into the
    directory ``root``.
    """
    ...

def build_index(root, base_url): # -> Element:
    """
    Create a new data.xml index file, by combining the xml description
    files for various packages and collections.  ``root`` should be the
    path to a directory containing the package xml and zip files; and
    the collection xml files.  The ``root`` directory is expected to
    have the following subdirectories::

      root/
        packages/ .................. subdirectory for packages
          corpora/ ................. zip & xml files for corpora
          grammars/ ................ zip & xml files for grammars
          taggers/ ................. zip & xml files for taggers
          tokenizers/ .............. zip & xml files for tokenizers
          etc.
        collections/ ............... xml files for collections

    For each package, there should be two files: ``package.zip``
    (where *package* is the package name)
    which contains the package itself as a compressed zip file; and
    ``package.xml``, which is an xml description of the package.  The
    zipfile ``package.zip`` should expand to a single subdirectory
    named ``package/``.  The base filename ``package`` must match
    the identifier given in the package's xml file.

    For each collection, there should be a single file ``collection.zip``
    describing the collection, where *collection* is the name of the collection.

    All identifiers (for both packages and collections) must be unique.
    """
    ...

_downloader = ...
download = ...
def download_shell(): # -> None:
    ...

def download_gui(): # -> None:
    ...

def update(): # -> None:
    ...

if __name__ == "__main__":
    parser = ...
    downloader = ...
