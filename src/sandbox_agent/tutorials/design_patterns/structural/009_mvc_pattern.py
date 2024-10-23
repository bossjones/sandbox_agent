# pragma: exclude file
"""
Model-View-Controller (MVC)

Overview of the MVC Pattern

The Model-View-Controller (MVC) is a design pattern widely used in web application development. It provides a
structure that separates an application's data, user interface, and control logic into three distinct components.
This separation allows for modular development and promotes organized, flexible, and scalable code.

1. Model: This represents the application's data and the business rules that govern access to and updates of this
   data. Often the model will also contain the logic to retrieve data from a database and to save data back to a
   database.
2. View: The view renders the contents of a model. It specifies exactly how the application's data should be
   presented. It is the view's responsibility to maintain consistency in its presentation when the model changes.
3. Controller: The controller interprets the inputs from the user, converting them into commands for the model or
   view. The controller depends on the view and the model. In some cases, the controller and the view are merged into
   a single component, notably in the Model-View-Presenter (MVP) pattern.

Implementing the MVC Pattern in Python

Consider an example where we are creating a simple web application to display and edit user information.
"""

from __future__ import annotations


class User:
    def __init__(self, name: str, age: int):
        """
        Initialize the User model with the given name and age.

        Args:
            name (str): The name of the user.
            age (int): The age of the user.
        """
        self.name = name
        self.age = age


class UserView:
    def display_user(self, user: User) -> None:
        """
        Display the user's information.

        Args:
            user (User): The user to display.
        """
        print(f"User: {user.name}, Age: {user.age}")


class UserController:
    def __init__(self, user: User, view: UserView):
        """
        Initialize the UserController with the given User model and UserView.

        Args:
            user (User): The User model.
            view (UserView): The UserView.
        """
        self.user = user
        self.view = view

    def set_user_name(self, name: str) -> None:
        """
        Set the user's name.

        Args:
            name (str): The new name for the user.
        """
        self.user.name = name

    def get_user_name(self) -> str:
        """
        Get the user's name.

        Returns:
            str: The user's name.
        """
        return self.user.name

    def set_user_age(self, age: int) -> None:
        """
        Set the user's age.

        Args:
            age (int): The new age for the user.
        """
        self.user.age = age

    def get_user_age(self) -> int:
        """
        Get the user's age.

        Returns:
            int: The user's age.
        """
        return self.user.age

    def update_view(self) -> None:
        """
        Update the view with the current user's information.
        """
        self.view.display_user(self.user)


# In this Python code, we have defined our model (User), view (UserView), and controller (UserController). The
# UserController is initialized with a User (model) and a UserView (view). The UserController can modify the User data
# and can update the UserView.

# ---

# Let me know if you need further clarification!
