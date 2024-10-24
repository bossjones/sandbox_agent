"""
This type stub file was generated by pyright.
"""

"""
Tools for reading and writing dependency trees.
The input is assumed to be in Malt-TAB format
(https://stp.lingfil.uu.se/~nivre/research/MaltXML.html).
"""
class DependencyGraph:
    """
    A container for the nodes and labelled edges of a dependency structure.
    """
    def __init__(self, tree_str=..., cell_extractor=..., zero_based=..., cell_separator=..., top_relation_label=...) -> None:
        """Dependency graph.

        We place a dummy `TOP` node with the index 0, since the root node is
        often assigned 0 as its head. This also means that the indexing of the
        nodes corresponds directly to the Malt-TAB format, which starts at 1.

        If zero-based is True, then Malt-TAB-like input with node numbers
        starting at 0 and the root node assigned -1 (as produced by, e.g.,
        zpar).

        :param str cell_separator: the cell separator. If not provided, cells
            are split by whitespace.

        :param str top_relation_label: the label by which the top relation is
            identified, for examlple, `ROOT`, `null` or `TOP`.
        """
        ...

    def remove_by_address(self, address): # -> None:
        """
        Removes the node with the given address.  References
        to this node in others will still exist.
        """
        ...

    def redirect_arcs(self, originals, redirect): # -> None:
        """
        Redirects arcs to any of the nodes in the originals list
        to the redirect node address.
        """
        ...

    def add_arc(self, head_address, mod_address): # -> None:
        """
        Adds an arc from the node specified by head_address to the
        node specified by the mod address.
        """
        ...

    def connect_graph(self): # -> None:
        """
        Fully connects all non-root nodes.  All nodes are set to be dependents
        of the root node.
        """
        ...

    def get_by_address(self, node_address): # -> dict[str, defaultdict[Any, list] | None]:
        """Return the node with the given address."""
        ...

    def contains_address(self, node_address): # -> bool:
        """
        Returns true if the graph contains a node with the given node
        address, false otherwise.
        """
        ...

    def to_dot(self): # -> str:
        """Return a dot representation suitable for using with Graphviz.

        >>> dg = DependencyGraph(
        ...     'John N 2\\n'
        ...     'loves V 0\\n'
        ...     'Mary N 2'
        ... )
        >>> print(dg.to_dot())
        digraph G{
        edge [dir=forward]
        node [shape=plaintext]
        <BLANKLINE>
        0 [label="0 (None)"]
        0 -> 2 [label="ROOT"]
        1 [label="1 (John)"]
        2 [label="2 (loves)"]
        2 -> 1 [label=""]
        2 -> 3 [label=""]
        3 [label="3 (Mary)"]
        }

        """
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self): # -> str:
        ...

    @staticmethod
    def load(filename, zero_based=..., cell_separator=..., top_relation_label=...): # -> list[DependencyGraph]:
        """
        :param filename: a name of a file in Malt-TAB format
        :param zero_based: nodes in the input file are numbered starting from 0
            rather than 1 (as produced by, e.g., zpar)
        :param str cell_separator: the cell separator. If not provided, cells
            are split by whitespace.
        :param str top_relation_label: the label by which the top relation is
            identified, for examlple, `ROOT`, `null` or `TOP`.

        :return: a list of DependencyGraphs

        """
        ...

    def left_children(self, node_index): # -> int:
        """
        Returns the number of left children under the node specified
        by the given address.
        """
        ...

    def right_children(self, node_index): # -> int:
        """
        Returns the number of right children under the node specified
        by the given address.
        """
        ...

    def add_node(self, node): # -> None:
        ...

    def tree(self): # -> Tree:
        """
        Starting with the ``root`` node, build a dependency tree using the NLTK
        ``Tree`` constructor. Dependency labels are omitted.
        """
        ...

    def triples(self, node=...): # -> Generator[tuple[tuple[Any | defaultdict[Any, list] | None, Any | defaultdict[Any, list] | None], defaultdict[Any, list] | None, tuple[defaultdict[Any, list] | None, defaultdict[Any, list] | None]] | Any, Any, None]:
        """
        Extract dependency triples of the form:
        ((head word, head tag), rel, (dep word, dep tag))
        """
        ...

    def contains_cycle(self): # -> list | Literal[False]:
        """Check whether there are cycles.

        >>> dg = DependencyGraph(treebank_data)
        >>> dg.contains_cycle()
        False

        >>> cyclic_dg = DependencyGraph()
        >>> top = {'word': None, 'deps': [1], 'rel': 'TOP', 'address': 0}
        >>> child1 = {'word': None, 'deps': [2], 'rel': 'NTOP', 'address': 1}
        >>> child2 = {'word': None, 'deps': [4], 'rel': 'NTOP', 'address': 2}
        >>> child3 = {'word': None, 'deps': [1], 'rel': 'NTOP', 'address': 3}
        >>> child4 = {'word': None, 'deps': [3], 'rel': 'NTOP', 'address': 4}
        >>> cyclic_dg.nodes = {
        ...     0: top,
        ...     1: child1,
        ...     2: child2,
        ...     3: child3,
        ...     4: child4,
        ... }
        >>> cyclic_dg.root = top

        >>> cyclic_dg.contains_cycle()
        [1, 2, 4, 3]

        """
        ...

    def get_cycle_path(self, curr_node, goal_node_index): # -> list:
        ...

    def to_conll(self, style): # -> str:
        """
        The dependency graph in CoNLL format.

        :param style: the style to use for the format (3, 4, 10 columns)
        :type style: int
        :rtype: str
        """
        ...

    def nx_graph(self): # -> MultiDiGraph:
        """Convert the data in a ``nodelist`` into a networkx labeled directed graph."""
        ...



def dot2img(dot_string, t=...): # -> str | bytes:
    """
    Create image representation fom dot_string, using the 'dot' program
    from the Graphviz package.

    Use the 't' argument to specify the image file format, for ex. 'jpeg', 'eps',
    'json', 'png' or 'webp' (Running 'dot -T:' lists all available formats).

    Note that the "capture_output" option of subprocess.run() is only available
    with text formats (like svg), but not with binary image formats (like png).
    """
    ...

class DependencyGraphError(Exception):
    """Dependency graph exception."""
    ...


def demo(): # -> None:
    ...

def malt_demo(nx=...): # -> None:
    """
    A demonstration of the result of reading a dependency
    version of the first sentence of the Penn Treebank.
    """
    ...

def conll_demo(): # -> None:
    """
    A demonstration of how to read a string representation of
    a CoNLL format dependency tree.
    """
    ...

def conll_file_demo(): # -> None:
    ...

def cycle_finding_demo(): # -> None:
    ...

treebank_data = ...
conll_data1 = ...
conll_data2 = ...
if __name__ == "__main__":
    ...