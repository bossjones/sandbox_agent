# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Flyweight Pattern

Overview of the Flyweight Pattern

The Flyweight pattern is a design pattern that minimizes memory use by sharing as much data as possible with similar objects. It is a way to use objects in large numbers when a simple repeated representation would use an unacceptable amount of memory. The concept is named after the idea of a flyweight, a boxing weight class that includes fighters who weigh very little.
"""

# Here is the extracted text from the image:

# ---

# **Flyweight Pattern**

# **Overview of the Flyweight Pattern**

# The Flyweight pattern is a design pattern that minimizes memory use by sharing as much data as possible with similar objects. It is a way to use objects in large numbers when a simple repeated representation would use an unacceptable amount of memory. The concept is named after the idea of a flyweight, a boxing weight class that includes fighters who weigh very little.

# **Implementing the Flyweight Pattern in Python**

# Let's consider an example where we create text formatting for a document editor. Here, each character in the document could be an object and have a character style (font size, bold, italic, etc.). But instead of applying the style to each character individually, we can use the Flyweight pattern to share the style between characters.

# Here's how you can implement the Flyweight pattern in Python:
from __future__ import annotations


class CharacterStyle:
    _styles = dict()

    def __new__(cls, font_size, bold, italic):
        key = (font_size, bold, italic)
        if key not in cls._styles:
            cls._styles[key] = super().__new__(cls)
            cls._styles[key].font_size = font_size
            cls._styles[key].bold = bold
            cls._styles[key].italic = italic
        return cls._styles[key]

    def __repr__(self):
        return f"{self.font_size}px {'bold' if self.bold else ''} {'italic' if self.italic else ''}"  # pyright: ignore[reportAttributeAccessIssue]


style1 = CharacterStyle(12, True, False)
style2 = CharacterStyle(12, True, False)
style3 = CharacterStyle(14, False, True)

print(style1)  # Outputs: 12px bold
print(style2)  # Outputs: 12px bold
print(style3)  # Outputs: 14px italic

print(style1 is style2)  # Outputs: True
print(style1 is style3)  # Outputs: False


# In this example, **CharacterStyle** uses a dictionary (`_styles`) to store styles that have already been created. When you request a style, it first checks if the style already exists. If it does, it returns the existing style. If not, it creates a new style.

# The Flyweight pattern can save a significant amount of memory if your program uses a large number of objects that have part of their state in common. However, it may introduce a level of complexity to the system and, if overused, can make the code harder to understand and maintain.

# ---

# Let me know if you need further clarification!
