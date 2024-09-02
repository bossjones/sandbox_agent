"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, ClassVar, Iterator, Optional, Set, TYPE_CHECKING, Tuple
from .flags import BaseFlags, alias_flag_value, fill_with_flags, flag_value
from typing_extensions import Self

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
__all__ = ('Permissions', 'PermissionOverwrite')
if TYPE_CHECKING:
    ...
class permission_alias(alias_flag_value):
    alias: str
    ...


def make_permission_alias(alias: str) -> Callable[[Callable[[Any], int]], permission_alias]:
    ...

@fill_with_flags()
class Permissions(BaseFlags):
    """Wraps up the Discord permission value.

    The properties provided are two way. You can set and retrieve individual
    bits using the properties as if they were regular bools. This allows
    you to edit permissions.

    .. versionchanged:: 1.3
        You can now use keyword arguments to initialize :class:`Permissions`
        similar to :meth:`update`.

    .. container:: operations

        .. describe:: x == y

            Checks if two permissions are equal.
        .. describe:: x != y

            Checks if two permissions are not equal.
        .. describe:: x <= y

            Checks if a permission is a subset of another permission.
        .. describe:: x >= y

            Checks if a permission is a superset of another permission.
        .. describe:: x < y

             Checks if a permission is a strict subset of another permission.
        .. describe:: x > y

             Checks if a permission is a strict superset of another permission.

        .. describe:: x | y, x |= y

            Returns a Permissions instance with all enabled flags from
            both x and y.

            .. versionadded:: 2.0

        .. describe:: x & y, x &= y

            Returns a Permissions instance with only flags enabled on
            both x and y.

            .. versionadded:: 2.0

        .. describe:: x ^ y, x ^= y

            Returns a Permissions instance with only flags enabled on
            only one of x or y, not on both.

            .. versionadded:: 2.0

        .. describe:: ~x

            Returns a Permissions instance with all flags inverted from x.

            .. versionadded:: 2.0

        .. describe:: hash(x)

               Return the permission's hash.
        .. describe:: iter(x)

               Returns an iterator of ``(perm, value)`` pairs. This allows it
               to be, for example, constructed as a dict or a list of pairs.
               Note that aliases are not shown.

        .. describe:: bool(b)

            Returns whether the permissions object has any permissions set to ``True``.

            .. versionadded:: 2.0

    Attributes
    -----------
    value: :class:`int`
        The raw value. This value is a bit array field of a 53-bit integer
        representing the currently available permissions. You should query
        permissions via the properties rather than using this raw value.
    """
    __slots__ = ...
    def __init__(self, permissions: int = ..., **kwargs: bool) -> None:
        ...

    def is_subset(self, other: Permissions) -> bool:
        """Returns ``True`` if self has the same or fewer permissions as other."""
        ...

    def is_superset(self, other: Permissions) -> bool:
        """Returns ``True`` if self has the same or more permissions as other."""
        ...

    def is_strict_subset(self, other: Permissions) -> bool:
        """Returns ``True`` if the permissions on other are a strict subset of those on self."""
        ...

    def is_strict_superset(self, other: Permissions) -> bool:
        """Returns ``True`` if the permissions on other are a strict superset of those on self."""
        ...

    __le__ = ...
    __ge__ = ...
    __lt__ = ...
    __gt__ = ...
    @classmethod
    def none(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all
        permissions set to ``False``."""
        ...

    @classmethod
    def all(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all
        permissions set to ``True``.
        """
        ...

    @classmethod
    def all_channel(cls) -> Self:
        """A :class:`Permissions` with all channel-specific permissions set to
        ``True`` and the guild-specific ones set to ``False``. The guild-specific
        permissions are currently:

        - :attr:`manage_expressions`
        - :attr:`view_audit_log`
        - :attr:`view_guild_insights`
        - :attr:`manage_guild`
        - :attr:`change_nickname`
        - :attr:`manage_nicknames`
        - :attr:`kick_members`
        - :attr:`ban_members`
        - :attr:`administrator`
        - :attr:`create_expressions`

        .. versionchanged:: 1.7
           Added :attr:`stream`, :attr:`priority_speaker` and :attr:`use_application_commands` permissions.

        .. versionchanged:: 2.0
           Added :attr:`create_public_threads`, :attr:`create_private_threads`, :attr:`manage_threads`,
           :attr:`use_external_stickers`, :attr:`send_messages_in_threads` and
           :attr:`request_to_speak` permissions.

        .. versionchanged:: 2.3
           Added :attr:`use_soundboard`, :attr:`create_expressions` permissions.
        """
        ...

    @classmethod
    def general(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all
        "General" permissions from the official Discord UI set to ``True``.

        .. versionchanged:: 1.7
           Permission :attr:`read_messages` is now included in the general permissions, but
           permissions :attr:`administrator`, :attr:`create_instant_invite`, :attr:`kick_members`,
           :attr:`ban_members`, :attr:`change_nickname` and :attr:`manage_nicknames` are
           no longer part of the general permissions.

        .. versionchanged:: 2.3
            Added :attr:`create_expressions` permission.
        """
        ...

    @classmethod
    def membership(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all
        "Membership" permissions from the official Discord UI set to ``True``.

        .. versionadded:: 1.7
        """
        ...

    @classmethod
    def text(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all
        "Text" permissions from the official Discord UI set to ``True``.

        .. versionchanged:: 1.7
           Permission :attr:`read_messages` is no longer part of the text permissions.
           Added :attr:`use_application_commands` permission.

        .. versionchanged:: 2.0
           Added :attr:`create_public_threads`, :attr:`create_private_threads`, :attr:`manage_threads`,
           :attr:`send_messages_in_threads` and :attr:`use_external_stickers` permissions.

        .. versionchanged:: 2.3
            Added :attr:`send_voice_messages` permission.
        """
        ...

    @classmethod
    def voice(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all
        "Voice" permissions from the official Discord UI set to ``True``."""
        ...

    @classmethod
    def stage(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all
        "Stage Channel" permissions from the official Discord UI set to ``True``.

        .. versionadded:: 1.7
        """
        ...

    @classmethod
    def stage_moderator(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all permissions
        for stage moderators set to ``True``. These permissions are currently:

        - :attr:`manage_channels`
        - :attr:`mute_members`
        - :attr:`move_members`

        .. versionadded:: 1.7

        .. versionchanged:: 2.0
            Added :attr:`manage_channels` permission and removed :attr:`request_to_speak` permission.
        """
        ...

    @classmethod
    def elevated(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all permissions
        that require 2FA set to ``True``. These permissions are currently:

        - :attr:`kick_members`
        - :attr:`ban_members`
        - :attr:`administrator`
        - :attr:`manage_channels`
        - :attr:`manage_guild`
        - :attr:`manage_messages`
        - :attr:`manage_roles`
        - :attr:`manage_webhooks`
        - :attr:`manage_expressions`
        - :attr:`manage_threads`
        - :attr:`moderate_members`

        .. versionadded:: 2.0
        """
        ...

    @classmethod
    def advanced(cls) -> Self:
        """A factory method that creates a :class:`Permissions` with all
        "Advanced" permissions from the official Discord UI set to ``True``.

        .. versionadded:: 1.7
        """
        ...

    def update(self, **kwargs: bool) -> None:
        r"""Bulk updates this permission object.

        Allows you to set multiple attributes by using keyword
        arguments. The names must be equivalent to the properties
        listed. Extraneous key/value pairs will be silently ignored.

        Parameters
        ------------
        \*\*kwargs
            A list of key/value pairs to bulk update permissions with.
        """
        ...

    def handle_overwrite(self, allow: int, deny: int) -> None:
        ...

    @flag_value
    def create_instant_invite(self) -> int:
        """:class:`bool`: Returns ``True`` if the user can create instant invites."""
        ...

    @flag_value
    def kick_members(self) -> int:
        """:class:`bool`: Returns ``True`` if the user can kick users from the guild."""
        ...

    @flag_value
    def ban_members(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can ban users from the guild."""
        ...

    @flag_value
    def administrator(self) -> int:
        """:class:`bool`: Returns ``True`` if a user is an administrator. This role overrides all other permissions.

        This also bypasses all channel-specific overrides.
        """
        ...

    @flag_value
    def manage_channels(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can edit, delete, or create channels in the guild.

        This also corresponds to the "Manage Channel" channel-specific override."""
        ...

    @flag_value
    def manage_guild(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can edit guild properties."""
        ...

    @flag_value
    def add_reactions(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can add reactions to messages."""
        ...

    @flag_value
    def view_audit_log(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can view the guild's audit log."""
        ...

    @flag_value
    def priority_speaker(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can be more easily heard while talking."""
        ...

    @flag_value
    def stream(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can stream in a voice channel."""
        ...

    @flag_value
    def read_messages(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can read messages from all or specific text channels."""
        ...

    @make_permission_alias('read_messages')
    def view_channel(self) -> int:
        """:class:`bool`: An alias for :attr:`read_messages`.

        .. versionadded:: 1.3
        """
        ...

    @flag_value
    def send_messages(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can send messages from all or specific text channels."""
        ...

    @flag_value
    def send_tts_messages(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can send TTS messages from all or specific text channels."""
        ...

    @flag_value
    def manage_messages(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can delete or pin messages in a text channel.

        .. note::

            Note that there are currently no ways to edit other people's messages.
        """
        ...

    @flag_value
    def embed_links(self) -> int:
        """:class:`bool`: Returns ``True`` if a user's messages will automatically be embedded by Discord."""
        ...

    @flag_value
    def attach_files(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can send files in their messages."""
        ...

    @flag_value
    def read_message_history(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can read a text channel's previous messages."""
        ...

    @flag_value
    def mention_everyone(self) -> int:
        """:class:`bool`: Returns ``True`` if a user's @everyone or @here will mention everyone in the text channel."""
        ...

    @flag_value
    def external_emojis(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can use emojis from other guilds."""
        ...

    @make_permission_alias('external_emojis')
    def use_external_emojis(self) -> int:
        """:class:`bool`: An alias for :attr:`external_emojis`.

        .. versionadded:: 1.3
        """
        ...

    @flag_value
    def view_guild_insights(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can view the guild's insights.

        .. versionadded:: 1.3
        """
        ...

    @flag_value
    def connect(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can connect to a voice channel."""
        ...

    @flag_value
    def speak(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can speak in a voice channel."""
        ...

    @flag_value
    def mute_members(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can mute other users."""
        ...

    @flag_value
    def deafen_members(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can deafen other users."""
        ...

    @flag_value
    def move_members(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can move users between other voice channels."""
        ...

    @flag_value
    def use_voice_activation(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can use voice activation in voice channels."""
        ...

    @flag_value
    def change_nickname(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can change their nickname in the guild."""
        ...

    @flag_value
    def manage_nicknames(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can change other user's nickname in the guild."""
        ...

    @flag_value
    def manage_roles(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can create or edit roles less than their role's position.

        This also corresponds to the "Manage Permissions" channel-specific override.
        """
        ...

    @make_permission_alias('manage_roles')
    def manage_permissions(self) -> int:
        """:class:`bool`: An alias for :attr:`manage_roles`.

        .. versionadded:: 1.3
        """
        ...

    @flag_value
    def manage_webhooks(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can create, edit, or delete webhooks."""
        ...

    @flag_value
    def manage_expressions(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can edit or delete emojis, stickers, and soundboard sounds.

        .. versionadded:: 2.3
        """
        ...

    @make_permission_alias('manage_expressions')
    def manage_emojis(self) -> int:
        """:class:`bool`: An alias for :attr:`manage_expressions`."""
        ...

    @make_permission_alias('manage_expressions')
    def manage_emojis_and_stickers(self) -> int:
        """:class:`bool`: An alias for :attr:`manage_expressions`.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def use_application_commands(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can use slash commands.

        .. versionadded:: 1.7
        """
        ...

    @flag_value
    def request_to_speak(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can request to speak in a stage channel.

        .. versionadded:: 1.7
        """
        ...

    @flag_value
    def manage_events(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can manage guild events.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def manage_threads(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can manage threads.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def create_public_threads(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can create public threads.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def create_private_threads(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can create private threads.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def external_stickers(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can use stickers from other guilds.

        .. versionadded:: 2.0
        """
        ...

    @make_permission_alias('external_stickers')
    def use_external_stickers(self) -> int:
        """:class:`bool`: An alias for :attr:`external_stickers`.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def send_messages_in_threads(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can send messages in threads.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def use_embedded_activities(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can launch an embedded application in a Voice channel.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def moderate_members(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can time out other members.

        .. versionadded:: 2.0
        """
        ...

    @flag_value
    def use_soundboard(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can use the soundboard.

        .. versionadded:: 2.3
        """
        ...

    @flag_value
    def create_expressions(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can create emojis, stickers, and soundboard sounds.

        .. versionadded:: 2.3
        """
        ...

    @flag_value
    def use_external_sounds(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can use sounds from other guilds.

        .. versionadded:: 2.3
        """
        ...

    @flag_value
    def send_voice_messages(self) -> int:
        """:class:`bool`: Returns ``True`` if a user can send voice messages.

        .. versionadded:: 2.3
        """
        ...



@_augment_from_permissions
class PermissionOverwrite:
    r"""A type that is used to represent a channel specific permission.

    Unlike a regular :class:`Permissions`\, the default value of a
    permission is equivalent to ``None`` and not ``False``. Setting
    a value to ``False`` is **explicitly** denying that permission,
    while setting a value to ``True`` is **explicitly** allowing
    that permission.

    The values supported by this are the same as :class:`Permissions`
    with the added possibility of it being set to ``None``.

    .. container:: operations

        .. describe:: x == y

            Checks if two overwrites are equal.
        .. describe:: x != y

            Checks if two overwrites are not equal.
        .. describe:: iter(x)

           Returns an iterator of ``(perm, value)`` pairs. This allows it
           to be, for example, constructed as a dict or a list of pairs.
           Note that aliases are not shown.

    Parameters
    -----------
    \*\*kwargs
        Set the value of permissions by their name.
    """
    __slots__ = ...
    if TYPE_CHECKING:
        VALID_NAMES: ClassVar[Set[str]]
        PURE_FLAGS: ClassVar[Set[str]]
        create_instant_invite: Optional[bool]
        kick_members: Optional[bool]
        ban_members: Optional[bool]
        administrator: Optional[bool]
        manage_channels: Optional[bool]
        manage_guild: Optional[bool]
        add_reactions: Optional[bool]
        view_audit_log: Optional[bool]
        priority_speaker: Optional[bool]
        stream: Optional[bool]
        read_messages: Optional[bool]
        view_channel: Optional[bool]
        send_messages: Optional[bool]
        send_tts_messages: Optional[bool]
        manage_messages: Optional[bool]
        embed_links: Optional[bool]
        attach_files: Optional[bool]
        read_message_history: Optional[bool]
        mention_everyone: Optional[bool]
        external_emojis: Optional[bool]
        use_external_emojis: Optional[bool]
        view_guild_insights: Optional[bool]
        connect: Optional[bool]
        speak: Optional[bool]
        mute_members: Optional[bool]
        deafen_members: Optional[bool]
        move_members: Optional[bool]
        use_voice_activation: Optional[bool]
        change_nickname: Optional[bool]
        manage_nicknames: Optional[bool]
        manage_roles: Optional[bool]
        manage_permissions: Optional[bool]
        manage_webhooks: Optional[bool]
        manage_expressions: Optional[bool]
        manage_emojis: Optional[bool]
        manage_emojis_and_stickers: Optional[bool]
        use_application_commands: Optional[bool]
        request_to_speak: Optional[bool]
        manage_events: Optional[bool]
        manage_threads: Optional[bool]
        create_public_threads: Optional[bool]
        create_private_threads: Optional[bool]
        send_messages_in_threads: Optional[bool]
        external_stickers: Optional[bool]
        use_external_stickers: Optional[bool]
        use_embedded_activities: Optional[bool]
        moderate_members: Optional[bool]
        use_soundboard: Optional[bool]
        use_external_sounds: Optional[bool]
        send_voice_messages: Optional[bool]
        create_expressions: Optional[bool]
        ...
    def __init__(self, **kwargs: Optional[bool]) -> None:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def pair(self) -> Tuple[Permissions, Permissions]:
        """Tuple[:class:`Permissions`, :class:`Permissions`]: Returns the (allow, deny) pair from this overwrite."""
        ...

    @classmethod
    def from_pair(cls, allow: Permissions, deny: Permissions) -> Self:
        """Creates an overwrite from an allow/deny pair of :class:`Permissions`."""
        ...

    def is_empty(self) -> bool:
        """Checks if the permission overwrite is currently empty.

        An empty permission overwrite is one that has no overwrites set
        to ``True`` or ``False``.

        Returns
        -------
        :class:`bool`
            Indicates if the overwrite is empty.
        """
        ...

    def update(self, **kwargs: Optional[bool]) -> None:
        r"""Bulk updates this permission overwrite object.

        Allows you to set multiple attributes by using keyword
        arguments. The names must be equivalent to the properties
        listed. Extraneous key/value pairs will be silently ignored.

        Parameters
        ------------
        \*\*kwargs
            A list of key/value pairs to bulk update with.
        """
        ...

    def __iter__(self) -> Iterator[Tuple[str, Optional[bool]]]:
        ...

