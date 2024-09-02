"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
from .vision import VisionDataset

CATEGORIES_2021 = ...
DATASET_URLS = ...
DATASET_MD5 = ...
class INaturalist(VisionDataset):
    """`iNaturalist <https://github.com/visipedia/inat_comp>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where the image files are stored.
            This class does not require/use annotation files.
        version (string, optional): Which version of the dataset to download/use. One of
            '2017', '2018', '2019', '2021_train', '2021_train_mini', '2021_valid'.
            Default: `2021_train`.
        target_type (string or list, optional): Type of target to use, for 2021 versions, one of:

            - ``full``: the full category (species)
            - ``kingdom``: e.g. "Animalia"
            - ``phylum``: e.g. "Arthropoda"
            - ``class``: e.g. "Insecta"
            - ``order``: e.g. "Coleoptera"
            - ``family``: e.g. "Cleridae"
            - ``genus``: e.g. "Trichodes"

            for 2017-2019 versions, one of:

            - ``full``: the full (numeric) category
            - ``super``: the super category, e.g. "Amphibians"

            Can also be a list to output a tuple with all specified target types.
            Defaults to ``full``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root: Union[str, Path], version: str = ..., target_type: Union[List[str], str] = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ..., download: bool = ...) -> None:
        ...

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        ...

    def __len__(self) -> int:
        ...

    def category_name(self, category_type: str, category_id: int) -> str:
        """
        Args:
            category_type(str): one of "full", "kingdom", "phylum", "class", "order", "family", "genus" or "super"
            category_id(int): an index (class id) from this category

        Returns:
            the name of the category
        """
        ...

    def download(self) -> None:
        ...

