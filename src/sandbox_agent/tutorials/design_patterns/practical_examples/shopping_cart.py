"""
Implementing a Shopping Cart using the Composite Pattern

The Composite Pattern is a structural design pattern that allows you to compose objects into tree-like structures and then work with these structures as if they were individual objects. It's a good fit when dealing with part-whole hierarchies, like an online shopping cart, which may consist of different items, bundles, or even other carts.
"""

# Concept of Shopping Cart with Composite Pattern
"""
The main idea is to have a common interface for both individual items and composites. This way, we can treat both individual items and compositions of items uniformly.
"""

# Code Example in Python

# Here's a simplified example of a shopping cart using the Composite Pattern:
from __future__ import annotations

from abc import ABC, abstractmethod


# Define common interface for both elements and composites
class CartItem(ABC):
    @abstractmethod
    def get_price(self):
        pass


# Leaf
class Product(CartItem):
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def get_price(self):
        return self.price


# Composite
class Cart(CartItem):
    def __init__(self):
        self.items = []

    def add_item(self, item: CartItem):
        self.items.append(item)

    def remove_item(self, item: CartItem):
        self.items.remove(item)

    def get_price(self):
        return sum(item.get_price() for item in self.items)


if __name__ == "__main__":
    # Create products
    book = Product("Book", 15)
    pen = Product("Pen", 5)

    # Create a shopping cart and add products to it
    cart = Cart()
    cart.add_item(book)
    cart.add_item(pen)

    # Print the total price of the cart
    print(f"Total: {cart.get_price()}")  # Output: 20

"""
In this code:
- CartItem is an abstract base class that declares the get_price method, which will be implemented by both leaf and composite objects.
- Product is a leaf element that represents individual items in the shopping cart.
- Cart is a composite object that can store both individual items and other carts.

With this setup, we can add or remove items from the shopping cart, and compute the total price of all items in the cart in a uniform way. The Composite Pattern allows us to simplify client code, as it can treat individual items and compositions of items uniformly.
"""
