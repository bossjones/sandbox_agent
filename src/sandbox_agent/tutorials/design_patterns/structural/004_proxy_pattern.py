# pragma: exclude file
"""
Proxy Pattern

Overview of the Proxy Pattern

The Proxy pattern provides a surrogate or placeholder for another object to control access to it. The pattern
involves a class, called the proxy, which represents the functionality of another class. This kind of design pattern
comes under structural patterns.

There are several types of proxy patterns, including "virtual proxy", "remote proxy", "protection proxy", and "smart
reference proxy". Each of these serves a unique purpose and provides a different way of controlling or managing
access to the actual object or service.

Implementing the Proxy Pattern in Python

Let's consider an example of a "protection proxy" that controls access to sensitive information.
"""

from __future__ import annotations


class SensitiveInfo:
    def read_sensitive_info(self) -> str:
        """
        Read sensitive information.

        Returns:
            str: The sensitive information.
        """
        return "Here is some sensitive information."


class SensitiveInfoProxy:
    def __init__(self):
        """
        Initialize the SensitiveInfoProxy with an instance of SensitiveInfo and a password.
        """
        self._sensitive_info = SensitiveInfo()
        self._password = "password"

    def read_sensitive_info(self, password: str) -> str:
        """
        Read sensitive information if the provided password is correct.

        Args:
            password (str): The password to access the sensitive information.

        Returns:
            str: The sensitive information if the password is correct, otherwise an access denied message.
        """
        if password == self._password:
            return self._sensitive_info.read_sensitive_info()
        else:
            return "Access Denied!"


proxy = SensitiveInfoProxy()
print(proxy.read_sensitive_info("wrong_password"))  # Outputs: Access Denied!
print(proxy.read_sensitive_info("password"))  # Outputs: Here is some sensitive information


# In this example, **SensitiveInfoProxy** serves as a proxy for **SensitiveInfo**. It adds a layer of security by
# requiring a password to read sensitive information. Without the correct password, it denies access.

# The Proxy pattern allows you to do complex or heavy operations on demand, control access to an object, or add a
# layer of protection to real functionality. But it might introduce a level of complexity to the system and, if
# overused, can make the code hard to understand and maintain.

# ---

# Let me know if you need anything else!
