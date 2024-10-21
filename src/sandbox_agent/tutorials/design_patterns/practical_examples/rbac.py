# pragma: exclude file
"""
Building a Role-Based Access Control System using the Proxy Pattern

The Proxy Pattern is a structural design pattern that allows you to provide a substitute or placeholder for another object.
A proxy controls access to this original object, allowing you to perform something either before or after the request gets
through to the original object.

In the context of a role-based access control system, we can use a proxy to represent the system's resources, controlling
access based on the user's role.
"""

# Concept of Role-Based Access Control System with Proxy Pattern
"""
In the context of a role-based access control system, we can use a proxy to represent the system's resources, controlling
access based on the user's role.
"""

# Code Example in Python

# Here is a basic example of a role-based access control system implemented using the Proxy Pattern:
from __future__ import annotations


class Resource:
    """
    Original object providing some valuable service.
    """

    def operation(self) -> str:
        """
        Perform the resource operation.

        Returns:
            str: The result of the resource operation.
        """
        return "Resource operation"


class Proxy:
    """
    Proxy controlling access to the original object.
    """

    def __init__(self, resource: Resource, role: str) -> None:
        """
        Initialize the proxy with a resource and a role.

        Args:
            resource (Resource): The resource to control access to.
            role (str): The role required to access the resource.
        """
        self.resource = resource
        self.role = role

    def operation(self) -> str:
        """
        Perform the resource operation if the role is 'admin', otherwise deny access.

        Returns:
            str: The result of the resource operation or an access denied message.
        """
        if self.role == "admin":
            return self.resource.operation()
        else:
            return "Access denied"


# Usage
if __name__ == "__main__":
    resource = Resource()

    # Creating a proxy for an admin role user
    proxy = Proxy(resource, "admin")
    print(proxy.operation())  # Output: Resource operation

    # Creating a proxy for a non-admin role user
    proxy = Proxy(resource, "user")
    print(proxy.operation())  # Output: Access denied

"""
In this code:
- Resource is the original object that we want to protect.
- Proxy is the proxy object that controls access to the Resource. It checks the user's role before allowing them to perform the operation on the Resource.

With this setup, the Proxy Pattern enables us to control access to a system's resources, ensuring that only users with the correct roles can perform certain operations. This can enhance the security of your system, as it helps prevent unauthorized access to sensitive resources.
"""
