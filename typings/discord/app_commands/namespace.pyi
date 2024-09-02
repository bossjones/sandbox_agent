"""
This type stub file was generated by pyright.
"""

from typing import Any, Iterator, List, NamedTuple, TYPE_CHECKING, Tuple
from ..interactions import Interaction
from ..types.interactions import ApplicationCommandInteractionDataOption, ResolvedData

"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
if TYPE_CHECKING:
    ...
__all__ = ('Namespace', )
class ResolveKey(NamedTuple):
    id: str
    type: int
    @classmethod
    def any_with(cls, id: str) -> ResolveKey:
        ...

    def __eq__(self, o: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...



class Namespace:
    """An object that holds the parameters being passed to a command in a mostly raw state.

    This class is deliberately simple and just holds the option name and resolved value as a simple
    key-pair mapping. These attributes can be accessed using dot notation. For example, an option
    with the name of ``example`` can be accessed using ``ns.example``. If an attribute is not found,
    then ``None`` is returned rather than an attribute error.

    .. warning::

        The key names come from the raw Discord data, which means that if a parameter was renamed then the
        renamed key is used instead of the function parameter name.

    .. versionadded:: 2.0

    .. container:: operations

        .. describe:: x == y

            Checks if two namespaces are equal by checking if all attributes are equal.
        .. describe:: x != y

            Checks if two namespaces are not equal.
        .. describe:: x[key]

            Returns an attribute if it is found, otherwise raises
            a :exc:`KeyError`.
        .. describe:: key in x

            Checks if the attribute is in the namespace.
        .. describe:: iter(x)

           Returns an iterator of ``(name, value)`` pairs. This allows it
           to be, for example, constructed as a dict or a list of pairs.

    This namespace object converts resolved objects into their appropriate form depending on their
    type. Consult the table below for conversion information.

    +-------------------------------------------+-------------------------------------------------------------------------------+
    |                Option Type                |                                 Resolved Type                                 |
    +===========================================+===============================================================================+
    | :attr:`.AppCommandOptionType.string`      | :class:`str`                                                                  |
    +-------------------------------------------+-------------------------------------------------------------------------------+
    | :attr:`.AppCommandOptionType.integer`     | :class:`int`                                                                  |
    +-------------------------------------------+-------------------------------------------------------------------------------+
    | :attr:`.AppCommandOptionType.boolean`     | :class:`bool`                                                                 |
    +-------------------------------------------+-------------------------------------------------------------------------------+
    | :attr:`.AppCommandOptionType.number`      | :class:`float`                                                                |
    +-------------------------------------------+-------------------------------------------------------------------------------+
    | :attr:`.AppCommandOptionType.user`        | :class:`~discord.User` or :class:`~discord.Member`                            |
    +-------------------------------------------+-------------------------------------------------------------------------------+
    | :attr:`.AppCommandOptionType.channel`     | :class:`.AppCommandChannel` or :class:`.AppCommandThread`                     |
    +-------------------------------------------+-------------------------------------------------------------------------------+
    | :attr:`.AppCommandOptionType.role`        | :class:`~discord.Role`                                                        |
    +-------------------------------------------+-------------------------------------------------------------------------------+
    | :attr:`.AppCommandOptionType.mentionable` | :class:`~discord.User` or :class:`~discord.Member`, or :class:`~discord.Role` |
    +-------------------------------------------+-------------------------------------------------------------------------------+
    | :attr:`.AppCommandOptionType.attachment`  | :class:`~discord.Attachment`                                                  |
    +-------------------------------------------+-------------------------------------------------------------------------------+

    .. note::

        In autocomplete interactions, the namespace might not be validated or filled in. Discord does not
        send the resolved data as well, so this means that certain fields end up just as IDs rather than
        the resolved data. In these cases, a :class:`discord.Object` is returned instead.

        This is a Discord limitation.
    """
    def __init__(self, interaction: Interaction, resolved: ResolvedData, options: List[ApplicationCommandInteractionDataOption]) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __getitem__(self, key: str) -> Any:
        ...

    def __contains__(self, key: str) -> Any:
        ...

    def __getattr__(self, attr: str) -> Any:
        ...

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        ...

