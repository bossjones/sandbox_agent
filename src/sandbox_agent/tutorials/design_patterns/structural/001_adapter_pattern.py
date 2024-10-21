# pragma: exclude file
"""
Overview of the Adapter Pattern

The Adapter pattern is a structural design pattern that allows objects with incompatible interfaces to work together.
This is achieved by creating a separate adapter class that converts the (incompatible) interface of a class (adaptee)
into another interface that clients require.

Implementing the Adapter Pattern in Python

Consider an example where we have a TextProcessor class that processes text. However, we want to use this
TextProcessor to process data from an XML source. The XML interface is incompatible with our TextProcessor. To solve
this, we could create an XMLAdapter to adapt the XML to our TextProcessor.

Envall, Henrik. Python Design Patterns: A Very Practical Guide (p. 23). Kindle Edition.
"""

from __future__ import annotations


class TextProcessor:
    def process(self, text: str) -> str:
        """
        Process the given text by converting it to uppercase.

        Args:
            text (str): The text to be processed.

        Returns:
            str: The processed text in uppercase.
        """
        return text.upper()


class XMLData:
    def get_data(self) -> str:
        """
        Get the XML data.

        Returns:
            str: The XML data as a string.
        """
        return "<xml>Hello, world!</xml>"


class XMLAdapter:
    def __init__(self, xml: XMLData):
        """
        Initialize the XMLAdapter with the given XMLData object.

        Args:
            xml (XMLData): The XMLData object to be adapted.
        """
        self.xml = xml

    def process(self) -> str:
        """
        Process the XML data by removing XML tags and using the TextProcessor.

        Returns:
            str: The processed text in uppercase.
        """
        text = self.xml.get_data()
        text = text.replace("<xml>", "").replace("</xml>", "")  # remove XML tags
        return TextProcessor().process(text)


xml_data = XMLData()
adapter = XMLAdapter(xml_data)
print(adapter.process())  # Outputs: HELLO, WORLD!


# In this code, XMLAdapter serves as an adapter for XMLData. The adapter takes XML data, removes the XML tags, and uses
# TextProcessor to process the cleaned text. Therefore, even though XMLData has a different interface than TextProcessor,
# we can still use it to process text, thanks to the adapter. The Adapter pattern allows you to reuse existing code
# without changing it, as the adapter can adapt interfaces without changing the existing code. This is especially useful
# when you want to integrate third-party or legacy code into your application, but the interfaces of the third-party or
# legacy code are not compatible with the rest of your application.
