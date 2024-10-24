"""
This type stub file was generated by pyright.
"""

from functools import total_ordering
from nltk.sem.logic import SubstituteBindingsI

"""
Basic data classes for representing feature structures, and for
performing basic operations on those feature structures.  A feature
structure is a mapping from feature identifiers to feature values,
where each feature value is either a basic value (such as a string or
an integer), or a nested feature structure.  There are two types of
feature structure, implemented by two subclasses of ``FeatStruct``:

    - feature dictionaries, implemented by ``FeatDict``, act like
      Python dictionaries.  Feature identifiers may be strings or
      instances of the ``Feature`` class.
    - feature lists, implemented by ``FeatList``, act like Python
      lists.  Feature identifiers are integers.

Feature structures are typically used to represent partial information
about objects.  A feature identifier that is not mapped to a value
stands for a feature whose value is unknown (*not* a feature without
a value).  Two feature structures that represent (potentially
overlapping) information about the same object can be combined by
unification.  When two inconsistent feature structures are unified,
the unification fails and returns None.

Features can be specified using "feature paths", or tuples of feature
identifiers that specify path through the nested feature structures to
a value.  Feature structures may contain reentrant feature values.  A
"reentrant feature value" is a single feature value that can be
accessed via multiple feature paths.  Unification preserves the
reentrance relations imposed by both of the unified feature
structures.  In the feature structure resulting from unification, any
modifications to a reentrant feature value will be visible using any
of its feature paths.

Feature structure variables are encoded using the ``nltk.sem.Variable``
class.  The variables' values are tracked using a bindings
dictionary, which maps variables to their values.  When two feature
structures are unified, a fresh bindings dictionary is created to
track their values; and before unification completes, all bound
variables are replaced by their values.  Thus, the bindings
dictionaries are usually strictly internal to the unification process.
However, it is possible to track the bindings of variables if you
choose to, by supplying your own initial bindings dictionary to the
``unify()`` function.

When unbound variables are unified with one another, they become
aliased.  This is encoded by binding one variable to the other.

Lightweight Feature Structures
==============================
Many of the functions defined by ``nltk.featstruct`` can be applied
directly to simple Python dictionaries and lists, rather than to
full-fledged ``FeatDict`` and ``FeatList`` objects.  In other words,
Python ``dicts`` and ``lists`` can be used as "light-weight" feature
structures.

    >>> from nltk.featstruct import unify
    >>> unify(dict(x=1, y=dict()), dict(a='a', y=dict(b='b')))  # doctest: +SKIP
    {'y': {'b': 'b'}, 'x': 1, 'a': 'a'}

However, you should keep in mind the following caveats:

  - Python dictionaries & lists ignore reentrance when checking for
    equality between values.  But two FeatStructs with different
    reentrances are considered nonequal, even if all their base
    values are equal.

  - FeatStructs can be easily frozen, allowing them to be used as
    keys in hash tables.  Python dictionaries and lists can not.

  - FeatStructs display reentrance in their string representations;
    Python dictionaries and lists do not.

  - FeatStructs may *not* be mixed with Python dictionaries and lists
    (e.g., when performing unification).

  - FeatStructs provide a number of useful methods, such as ``walk()``
    and ``cyclic()``, which are not available for Python dicts and lists.

In general, if your feature structures will contain any reentrances,
or if you plan to use them as dictionary keys, it is strongly
recommended that you use full-fledged ``FeatStruct`` objects.
"""
@total_ordering
class FeatStruct(SubstituteBindingsI):
    """
    A mapping from feature identifiers to feature values, where each
    feature value is either a basic value (such as a string or an
    integer), or a nested feature structure.  There are two types of
    feature structure:

      - feature dictionaries, implemented by ``FeatDict``, act like
        Python dictionaries.  Feature identifiers may be strings or
        instances of the ``Feature`` class.
      - feature lists, implemented by ``FeatList``, act like Python
        lists.  Feature identifiers are integers.

    Feature structures may be indexed using either simple feature
    identifiers or 'feature paths.'  A feature path is a sequence
    of feature identifiers that stand for a corresponding sequence of
    indexing operations.  In particular, ``fstruct[(f1,f2,...,fn)]`` is
    equivalent to ``fstruct[f1][f2]...[fn]``.

    Feature structures may contain reentrant feature structures.  A
    "reentrant feature structure" is a single feature structure
    object that can be accessed via multiple feature paths.  Feature
    structures may also be cyclic.  A feature structure is "cyclic"
    if there is any feature path from the feature structure to itself.

    Two feature structures are considered equal if they assign the
    same values to all features, and have the same reentrancies.

    By default, feature structures are mutable.  They may be made
    immutable with the ``freeze()`` method.  Once they have been
    frozen, they may be hashed, and thus used as dictionary keys.
    """
    _frozen = ...
    def __new__(cls, features=..., **morefeatures):
        """
        Construct and return a new feature structure.  If this
        constructor is called directly, then the returned feature
        structure will be an instance of either the ``FeatDict`` class
        or the ``FeatList`` class.

        :param features: The initial feature values for this feature
            structure:

            - FeatStruct(string) -> FeatStructReader().read(string)
            - FeatStruct(mapping) -> FeatDict(mapping)
            - FeatStruct(sequence) -> FeatList(sequence)
            - FeatStruct() -> FeatDict()
        :param morefeatures: If ``features`` is a mapping or None,
            then ``morefeatures`` provides additional features for the
            ``FeatDict`` constructor.
        """
        ...

    def equal_values(self, other, check_reentrance=...): # -> bool:
        """
        Return True if ``self`` and ``other`` assign the same value to
        to every feature.  In particular, return true if
        ``self[p]==other[p]`` for every feature path *p* such
        that ``self[p]`` or ``other[p]`` is a base value (i.e.,
        not a nested feature structure).

        :param check_reentrance: If True, then also return False if
            there is any difference between the reentrances of ``self``
            and ``other``.
        :note: the ``==`` is equivalent to ``equal_values()`` with
            ``check_reentrance=True``.
        """
        ...

    def __eq__(self, other) -> bool:
        """
        Return true if ``self`` and ``other`` are both feature structures,
        assign the same values to all features, and contain the same
        reentrances.  I.e., return
        ``self.equal_values(other, check_reentrance=True)``.

        :see: ``equal_values()``
        """
        ...

    def __ne__(self, other) -> bool:
        ...

    def __lt__(self, other) -> bool:
        ...

    def __hash__(self) -> int:
        """
        If this feature structure is frozen, return its hash value;
        otherwise, raise ``TypeError``.
        """
        ...

    _FROZEN_ERROR = ...
    def freeze(self): # -> None:
        """
        Make this feature structure, and any feature structures it
        contains, immutable.  Note: this method does not attempt to
        'freeze' any feature value that is not a ``FeatStruct``; it
        is recommended that you use only immutable feature values.
        """
        ...

    def frozen(self): # -> bool:
        """
        Return True if this feature structure is immutable.  Feature
        structures can be made immutable with the ``freeze()`` method.
        Immutable feature structures may not be made mutable again,
        but new mutable copies can be produced with the ``copy()`` method.
        """
        ...

    def copy(self, deep=...): # -> Self:
        """
        Return a new copy of ``self``.  The new copy will not be frozen.

        :param deep: If true, create a deep copy; if false, create
            a shallow copy.
        """
        ...

    def __deepcopy__(self, memo):
        ...

    def cyclic(self):
        """
        Return True if this feature structure contains itself.
        """
        ...

    def walk(self): # -> Generator[Self | Any, Any, None]:
        """
        Return an iterator that generates this feature structure, and
        each feature structure it contains.  Each feature structure will
        be generated exactly once.
        """
        ...

    def substitute_bindings(self, bindings):
        """:see: ``nltk.featstruct.substitute_bindings()``"""
        ...

    def retract_bindings(self, bindings):
        """:see: ``nltk.featstruct.retract_bindings()``"""
        ...

    def variables(self): # -> set | None:
        """:see: ``nltk.featstruct.find_variables()``"""
        ...

    def rename_variables(self, vars=..., used_vars=..., new_vars=...): # -> None:
        """:see: ``nltk.featstruct.rename_variables()``"""
        ...

    def remove_variables(self): # -> None:
        """
        Return the feature structure that is obtained by deleting
        any feature whose value is a ``Variable``.

        :rtype: FeatStruct
        """
        ...

    def unify(self, other, bindings=..., trace=..., fail=..., rename_vars=...): # -> _UnificationFailure | None:
        ...

    def subsumes(self, other):
        """
        Return True if ``self`` subsumes ``other``.  I.e., return true
        If unifying ``self`` with ``other`` would result in a feature
        structure equal to ``other``.
        """
        ...

    def __repr__(self):
        """
        Display a single-line representation of this feature structure,
        suitable for embedding in other representations.
        """
        ...



_FROZEN_ERROR = ...
_FROZEN_NOTICE = ...
class FeatDict(FeatStruct, dict):
    """
    A feature structure that acts like a Python dictionary.  I.e., a
    mapping from feature identifiers to feature values, where a feature
    identifier can be a string or a ``Feature``; and where a feature value
    can be either a basic value (such as a string or an integer), or a nested
    feature structure.  A feature identifiers for a ``FeatDict`` is
    sometimes called a "feature name".

    Two feature dicts are considered equal if they assign the same
    values to all features, and have the same reentrances.

    :see: ``FeatStruct`` for information about feature paths, reentrance,
        cyclic feature structures, mutability, freezing, and hashing.
    """
    def __init__(self, features=..., **morefeatures) -> None:
        """
        Create a new feature dictionary, with the specified features.

        :param features: The initial value for this feature
            dictionary.  If ``features`` is a ``FeatStruct``, then its
            features are copied (shallow copy).  If ``features`` is a
            dict, then a feature is created for each item, mapping its
            key to its value.  If ``features`` is a string, then it is
            processed using ``FeatStructReader``.  If ``features`` is a list of
            tuples ``(name, val)``, then a feature is created for each tuple.
        :param morefeatures: Additional features for the new feature
            dictionary.  If a feature is listed under both ``features`` and
            ``morefeatures``, then the value from ``morefeatures`` will be
            used.
        """
        ...

    _INDEX_ERROR = ...
    def __getitem__(self, name_or_path): # -> Self:
        """If the feature with the given name or path exists, return
        its value; otherwise, raise ``KeyError``."""
        ...

    def get(self, name_or_path, default=...): # -> Self | None:
        """If the feature with the given name or path exists, return its
        value; otherwise, return ``default``."""
        ...

    def __contains__(self, name_or_path): # -> bool:
        """Return true if a feature with the given name or path exists."""
        ...

    def has_key(self, name_or_path): # -> bool:
        """Return true if a feature with the given name or path exists."""
        ...

    def __delitem__(self, name_or_path): # -> None:
        """If the feature with the given name or path exists, delete
        its value; otherwise, raise ``KeyError``."""
        ...

    def __setitem__(self, name_or_path, value): # -> None:
        """Set the value for the feature with the given name or path
        to ``value``.  If ``name_or_path`` is an invalid path, raise
        ``KeyError``."""
        ...

    clear = ...
    pop = ...
    popitem = ...
    setdefault = ...
    def update(self, features=..., **morefeatures): # -> None:
        ...

    def __deepcopy__(self, memo): # -> Self:
        ...

    def __str__(self) -> str:
        """
        Display a multi-line representation of this feature dictionary
        as an FVM (feature value matrix).
        """
        ...



class FeatList(FeatStruct, list):
    """
    A list of feature values, where each feature value is either a
    basic value (such as a string or an integer), or a nested feature
    structure.

    Feature lists may contain reentrant feature values.  A "reentrant
    feature value" is a single feature value that can be accessed via
    multiple feature paths.  Feature lists may also be cyclic.

    Two feature lists are considered equal if they assign the same
    values to all features, and have the same reentrances.

    :see: ``FeatStruct`` for information about feature paths, reentrance,
        cyclic feature structures, mutability, freezing, and hashing.
    """
    def __init__(self, features=...) -> None:
        """
        Create a new feature list, with the specified features.

        :param features: The initial list of features for this feature
            list.  If ``features`` is a string, then it is paresd using
            ``FeatStructReader``.  Otherwise, it should be a sequence
            of basic values and nested feature structures.
        """
        ...

    _INDEX_ERROR = ...
    def __getitem__(self, name_or_path): # -> Self:
        ...

    def __delitem__(self, name_or_path): # -> None:
        """If the feature with the given name or path exists, delete
        its value; otherwise, raise ``KeyError``."""
        ...

    def __setitem__(self, name_or_path, value): # -> None:
        """Set the value for the feature with the given name or path
        to ``value``.  If ``name_or_path`` is an invalid path, raise
        ``KeyError``."""
        ...

    __iadd__ = ...
    __imul__ = ...
    append = ...
    extend = ...
    insert = ...
    pop = ...
    remove = ...
    reverse = ...
    sort = ...
    def __deepcopy__(self, memo): # -> Self:
        ...



def substitute_bindings(fstruct, bindings, fs_class=...):
    """
    Return the feature structure that is obtained by replacing each
    variable bound by ``bindings`` with its binding.  If a variable is
    aliased to a bound variable, then it will be replaced by that
    variable's value.  If a variable is aliased to an unbound
    variable, then it will be replaced by that variable.

    :type bindings: dict(Variable -> any)
    :param bindings: A dictionary mapping from variables to values.
    """
    ...

def retract_bindings(fstruct, bindings, fs_class=...):
    """
    Return the feature structure that is obtained by replacing each
    feature structure value that is bound by ``bindings`` with the
    variable that binds it.  A feature structure value must be
    identical to a bound value (i.e., have equal id) to be replaced.

    ``bindings`` is modified to point to this new feature structure,
    rather than the original feature structure.  Feature structure
    values in ``bindings`` may be modified if they are contained in
    ``fstruct``.
    """
    ...

def find_variables(fstruct, fs_class=...): # -> set | None:
    """
    :return: The set of variables used by this feature structure.
    :rtype: set(Variable)
    """
    ...

def rename_variables(fstruct, vars=..., used_vars=..., new_vars=..., fs_class=...): # -> None:
    """
    Return the feature structure that is obtained by replacing
    any of this feature structure's variables that are in ``vars``
    with new variables.  The names for these new variables will be
    names that are not used by any variable in ``vars``, or in
    ``used_vars``, or in this feature structure.

    :type vars: set
    :param vars: The set of variables that should be renamed.
        If not specified, ``find_variables(fstruct)`` is used; i.e., all
        variables will be given new names.
    :type used_vars: set
    :param used_vars: A set of variables whose names should not be
        used by the new variables.
    :type new_vars: dict(Variable -> Variable)
    :param new_vars: A dictionary that is used to hold the mapping
        from old variables to new variables.  For each variable *v*
        in this feature structure:

        - If ``new_vars`` maps *v* to *v'*, then *v* will be
          replaced by *v'*.
        - If ``new_vars`` does not contain *v*, but ``vars``
          does contain *v*, then a new entry will be added to
          ``new_vars``, mapping *v* to the new variable that is used
          to replace it.

    To consistently rename the variables in a set of feature
    structures, simply apply rename_variables to each one, using
    the same dictionary:

        >>> from nltk.featstruct import FeatStruct
        >>> fstruct1 = FeatStruct('[subj=[agr=[gender=?y]], obj=[agr=[gender=?y]]]')
        >>> fstruct2 = FeatStruct('[subj=[agr=[number=?z,gender=?y]], obj=[agr=[number=?z,gender=?y]]]')
        >>> new_vars = {}  # Maps old vars to alpha-renamed vars
        >>> fstruct1.rename_variables(new_vars=new_vars)
        [obj=[agr=[gender=?y2]], subj=[agr=[gender=?y2]]]
        >>> fstruct2.rename_variables(new_vars=new_vars)
        [obj=[agr=[gender=?y2, number=?z2]], subj=[agr=[gender=?y2, number=?z2]]]

    If new_vars is not specified, then an empty dictionary is used.
    """
    ...

def remove_variables(fstruct, fs_class=...): # -> None:
    """
    :rtype: FeatStruct
    :return: The feature structure that is obtained by deleting
        all features whose values are ``Variables``.
    """
    ...

class _UnificationFailure:
    def __repr__(self): # -> Literal['nltk.featstruct.UnificationFailure']:
        ...



UnificationFailure = ...
def unify(fstruct1, fstruct2, bindings=..., trace=..., fail=..., rename_vars=..., fs_class=...): # -> _UnificationFailure | None:
    """
    Unify ``fstruct1`` with ``fstruct2``, and return the resulting feature
    structure.  This unified feature structure is the minimal
    feature structure that contains all feature value assignments from both
    ``fstruct1`` and ``fstruct2``, and that preserves all reentrancies.

    If no such feature structure exists (because ``fstruct1`` and
    ``fstruct2`` specify incompatible values for some feature), then
    unification fails, and ``unify`` returns None.

    Bound variables are replaced by their values.  Aliased
    variables are replaced by their representative variable
    (if unbound) or the value of their representative variable
    (if bound).  I.e., if variable *v* is in ``bindings``,
    then *v* is replaced by ``bindings[v]``.  This will
    be repeated until the variable is replaced by an unbound
    variable or a non-variable value.

    Unbound variables are bound when they are unified with
    values; and aliased when they are unified with variables.
    I.e., if variable *v* is not in ``bindings``, and is
    unified with a variable or value *x*, then
    ``bindings[v]`` is set to *x*.

    If ``bindings`` is unspecified, then all variables are
    assumed to be unbound.  I.e., ``bindings`` defaults to an
    empty dict.

        >>> from nltk.featstruct import FeatStruct
        >>> FeatStruct('[a=?x]').unify(FeatStruct('[b=?x]'))
        [a=?x, b=?x2]

    :type bindings: dict(Variable -> any)
    :param bindings: A set of variable bindings to be used and
        updated during unification.
    :type trace: bool
    :param trace: If true, generate trace output.
    :type rename_vars: bool
    :param rename_vars: If True, then rename any variables in
        ``fstruct2`` that are also used in ``fstruct1``, in order to
        avoid collisions on variable names.
    """
    ...

class _UnificationFailureError(Exception):
    """An exception that is used by ``_destructively_unify`` to abort
    unification when a failure is encountered."""
    ...


def subsumes(fstruct1, fstruct2):
    """
    Return True if ``fstruct1`` subsumes ``fstruct2``.  I.e., return
    true if unifying ``fstruct1`` with ``fstruct2`` would result in a
    feature structure equal to ``fstruct2.``

    :rtype: bool
    """
    ...

def conflicts(fstruct1, fstruct2, trace=...): # -> list:
    """
    Return a list of the feature paths of all features which are
    assigned incompatible values by ``fstruct1`` and ``fstruct2``.

    :rtype: list(tuple)
    """
    ...

class SubstituteBindingsSequence(SubstituteBindingsI):
    """
    A mixin class for sequence classes that distributes variables() and
    substitute_bindings() over the object's elements.
    """
    def variables(self): # -> list[Variable]:
        ...

    def substitute_bindings(self, bindings): # -> Self:
        ...

    def subst(self, v, bindings):
        ...



class FeatureValueTuple(SubstituteBindingsSequence, tuple):
    """
    A base feature value that is a tuple of other base feature values.
    FeatureValueTuple implements ``SubstituteBindingsI``, so it any
    variable substitutions will be propagated to the elements
    contained by the set.  A ``FeatureValueTuple`` is immutable.
    """
    def __repr__(self): # -> str:
        ...



class FeatureValueSet(SubstituteBindingsSequence, frozenset):
    """
    A base feature value that is a set of other base feature values.
    FeatureValueSet implements ``SubstituteBindingsI``, so it any
    variable substitutions will be propagated to the elements
    contained by the set.  A ``FeatureValueSet`` is immutable.
    """
    def __repr__(self): # -> str:
        ...

    __str__ = ...


class FeatureValueUnion(SubstituteBindingsSequence, frozenset):
    """
    A base feature value that represents the union of two or more
    ``FeatureValueSet`` or ``Variable``.
    """
    def __new__(cls, values): # -> FeatureValueSet | Self:
        ...

    def __repr__(self): # -> str:
        ...



class FeatureValueConcat(SubstituteBindingsSequence, tuple):
    """
    A base feature value that represents the concatenation of two or
    more ``FeatureValueTuple`` or ``Variable``.
    """
    def __new__(cls, values): # -> FeatureValueTuple | Self:
        ...

    def __repr__(self): # -> str:
        ...



@total_ordering
class Feature:
    """
    A feature identifier that's specialized to put additional
    constraints, default values, etc.
    """
    def __init__(self, name, default=..., display=...) -> None:
        ...

    @property
    def name(self): # -> Any:
        """The name of this feature."""
        ...

    @property
    def default(self): # -> None:
        """Default value for this feature."""
        ...

    @property
    def display(self): # -> None:
        """Custom display location: can be prefix, or slash."""
        ...

    def __repr__(self): # -> LiteralString:
        ...

    def __lt__(self, other) -> bool:
        ...

    def __eq__(self, other) -> bool:
        ...

    def __ne__(self, other) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def read_value(self, s, position, reentrances, parser):
        ...

    def unify_base_values(self, fval1, fval2, bindings): # -> _UnificationFailure:
        """
        If possible, return a single value..  If not, return
        the value ``UnificationFailure``.
        """
        ...



class SlashFeature(Feature):
    def read_value(self, s, position, reentrances, parser):
        ...



class RangeFeature(Feature):
    RANGE_RE = ...
    def read_value(self, s, position, reentrances, parser): # -> tuple[tuple[int, int], int]:
        ...

    def unify_base_values(self, fval1, fval2, bindings): # -> _UnificationFailure | tuple:
        ...



SLASH = ...
TYPE = ...
@total_ordering
class CustomFeatureValue:
    """
    An abstract base class for base values that define a custom
    unification method.  The custom unification method of
    ``CustomFeatureValue`` will be used during unification if:

      - The ``CustomFeatureValue`` is unified with another base value.
      - The ``CustomFeatureValue`` is not the value of a customized
        ``Feature`` (which defines its own unification method).

    If two ``CustomFeatureValue`` objects are unified with one another
    during feature structure unification, then the unified base values
    they return *must* be equal; otherwise, an ``AssertionError`` will
    be raised.

    Subclasses must define ``unify()``, ``__eq__()`` and ``__lt__()``.
    Subclasses may also wish to define ``__hash__()``.
    """
    def unify(self, other):
        """
        If this base value unifies with ``other``, then return the
        unified value.  Otherwise, return ``UnificationFailure``.
        """
        ...

    def __eq__(self, other) -> bool:
        ...

    def __ne__(self, other) -> bool:
        ...

    def __lt__(self, other) -> bool:
        ...

    def __hash__(self) -> int:
        ...



class FeatStructReader:
    def __init__(self, features=..., fdict_class=..., flist_class=..., logic_parser=...) -> None:
        ...

    def fromstring(self, s, fstruct=...):
        """
        Convert a string representation of a feature structure (as
        displayed by repr) into a ``FeatStruct``.  This process
        imposes the following restrictions on the string
        representation:

        - Feature names cannot contain any of the following:
          whitespace, parentheses, quote marks, equals signs,
          dashes, commas, and square brackets.  Feature names may
          not begin with plus signs or minus signs.
        - Only the following basic feature value are supported:
          strings, integers, variables, None, and unquoted
          alphanumeric strings.
        - For reentrant values, the first mention must specify
          a reentrance identifier and a value; and any subsequent
          mentions must use arrows (``'->'``) to reference the
          reentrance identifier.
        """
        ...

    _START_FSTRUCT_RE = ...
    _END_FSTRUCT_RE = ...
    _SLASH_RE = ...
    _FEATURE_NAME_RE = ...
    _REENTRANCE_RE = ...
    _TARGET_RE = ...
    _ASSIGN_RE = ...
    _COMMA_RE = ...
    _BARE_PREFIX_RE = ...
    _START_FDICT_RE = ...
    def read_partial(self, s, position=..., reentrances=..., fstruct=...): # -> tuple[Any, Any] | tuple[Any, int] | None:
        """
        Helper function that reads in a feature structure.

        :param s: The string to read.
        :param position: The position in the string to start parsing.
        :param reentrances: A dictionary from reentrance ids to values.
            Defaults to an empty dictionary.
        :return: A tuple (val, pos) of the feature structure created by
            parsing and the position where the parsed feature structure ends.
        :rtype: bool
        """
        ...

    def read_value(self, s, position, reentrances): # -> Any:
        ...

    VALUE_HANDLERS = ...
    def read_fstruct_value(self, s, position, reentrances, match): # -> tuple[Any, Any] | tuple[Any, int] | None:
        ...

    def read_str_value(self, s, position, reentrances, match): # -> tuple[Any, int]:
        ...

    def read_int_value(self, s, position, reentrances, match): # -> tuple[int, Any]:
        ...

    def read_var_value(self, s, position, reentrances, match): # -> tuple[Variable, Any]:
        ...

    _SYM_CONSTS = ...
    def read_sym_value(self, s, position, reentrances, match): # -> tuple:
        ...

    def read_app_value(self, s, position, reentrances, match): # -> tuple[Any | AndExpression | IffExpression | ImpExpression | OrExpression, Any]:
        """Mainly included for backwards compat."""
        ...

    def read_logic_value(self, s, position, reentrances, match): # -> tuple[Any | AndExpression | IffExpression | ImpExpression | OrExpression, Any]:
        ...

    def read_tuple_value(self, s, position, reentrances, match): # -> tuple:
        ...

    def read_set_value(self, s, position, reentrances, match): # -> tuple:
        ...



def display_unification(fs1, fs2, indent=...):
    ...

def interactive_demo(trace=...): # -> None:
    ...

def demo(trace=...): # -> None:
    """
    Just for testing
    """
    ...

if __name__ == "__main__":
    ...
__all__ = ["FeatStruct", "FeatDict", "FeatList", "unify", "subsumes", "conflicts", "Feature", "SlashFeature", "RangeFeature", "SLASH", "TYPE", "FeatStructReader"]
