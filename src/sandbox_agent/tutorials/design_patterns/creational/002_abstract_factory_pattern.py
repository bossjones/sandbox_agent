# creational design patterns

"""
Abstract Factory Pattern

Overview of the Abstract Factory Pattern The Abstract Factory pattern is a design pattern that provides an interface for creating families of related or dependent objects without specifying their concrete classes. It is like a factory of factories and it is one level higher in abstraction than the Factory Pattern. This pattern can be beneficial when a system should be independent of how its products are created, composed, and represented, and when systems need to work with multiple families of products. Implementing the Abstract Factory Pattern in Python To illustrate the Abstract Factory pattern, let's use an example of a pet and pet food factory. There will be a DogFactory and a CatFactory, each producing a pet and its corresponding food.

Envall, Henrik. Python Design Patterns: A Very Practical Guide (p. 13). Kindle Edition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Pet(ABC):
    """Abstract base class representing a pet."""

    @abstractmethod
    def speak(self) -> str:
        """
        Make the pet speak.

        Returns:
            str: The sound the pet makes.
        """
        pass


class Dog(Pet):
    """Concrete class representing a dog."""

    def speak(self) -> str:
        """
        Make the dog speak.

        Returns:
            str: The sound the dog makes.
        """
        return "Woof!"


class Cat(Pet):
    """Concrete class representing a cat."""

    def speak(self) -> str:
        """
        Make the cat speak.

        Returns:
            str: The sound the cat makes.
        """
        return "Meow!"


class PetFood(ABC):
    """Abstract base class representing pet food."""

    @abstractmethod
    def taste(self) -> str:
        """
        Describe the taste of the pet food.

        Returns:
            str: The taste description of the pet food.
        """
        pass


class DogFood(PetFood):
    """Concrete class representing dog food."""

    def taste(self) -> str:
        """
        Describe the taste of the dog food.

        Returns:
            str: The taste description of the dog food.
        """
        return "Tastes like chicken!"


class CatFood(PetFood):
    """Concrete class representing cat food."""

    def taste(self) -> str:
        """
        Describe the taste of the cat food.

        Returns:
            str: The taste description of the cat food.
        """
        return "Tastes like tuna!"


class PetFactory(ABC):
    """Abstract base class representing a pet factory."""

    @abstractmethod
    def get_pet(self) -> Pet:
        """
        Get a pet instance.

        Returns:
            Pet: The pet instance.
        """
        pass

    @abstractmethod
    def get_pet_food(self) -> PetFood:
        """
        Get a pet food instance.

        Returns:
            PetFood: The pet food instance.
        """
        pass


class DogFactory(PetFactory):
    """Concrete class representing a dog factory."""

    def get_pet(self) -> Pet:
        """
        Get a dog instance.

        Returns:
            Pet: The dog instance.
        """
        return Dog()

    def get_pet_food(self) -> PetFood:
        """
        Get a dog food instance.

        Returns:
            PetFood: The dog food instance.
        """
        return DogFood()


class CatFactory(PetFactory):
    """Concrete class representing a cat factory."""

    def get_pet(self) -> Pet:
        """
        Get a cat instance.

        Returns:
            Pet: The cat instance.
        """
        return Cat()

    def get_pet_food(self) -> PetFood:
        """
        Get a cat food instance.

        Returns:
            PetFood: The cat food instance.
        """
        return CatFood()


# Now let's use our abstract factory
for pet_factory in [DogFactory(), CatFactory()]:
    pet = pet_factory.get_pet()
    pet_food = pet_factory.get_pet_food()
    print(f"Our pet is '{pet.speak()}'!")
    print(f"Its food is '{pet_food.taste()}'!")
