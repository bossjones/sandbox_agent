"""
This type stub file was generated by pyright.
"""

class FStructure(dict):
    def safeappend(self, key, item): # -> None:
        """
        Append 'item' to the list at 'key'.  If no list exists for 'key', then
        construct one.
        """
        ...

    def __setitem__(self, key, value): # -> None:
        ...

    def __getitem__(self, key):
        ...

    def __contains__(self, key): # -> bool:
        ...

    def to_glueformula_list(self, glue_dict):
        ...

    def to_depgraph(self, rel=...): # -> DependencyGraph:
        ...

    @staticmethod
    def read_depgraph(depgraph): # -> tuple | FStructure:
        ...

    def __repr__(self): # -> str:
        ...

    def __str__(self) -> str:
        ...

    def pretty_format(self, indent=...): # -> str:
        ...



def demo_read_depgraph(): # -> None:
    ...

if __name__ == "__main__":
    ...
