"""
This type stub file was generated by pyright.
"""

from pinecone.core.client.model_utils import ModelNormal, cached_property, convert_js_args_to_python_args

"""
    Pinecone Control Plane API

    Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

    The version of the OpenAPI document: v1
    Contact: support@pinecone.io
    Generated by: https://openapi-generator.tech
"""
class CreateCollectionRequest(ModelNormal):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    Attributes:
      allowed_values (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          with a capitalized key describing the allowed value and an allowed
          value. These dicts store the allowed enum values.
      attribute_map (dict): The key is attribute name
          and the value is json key in definition.
      discriminator_value_class_map (dict): A dict to go from the discriminator
          variable value to the discriminator class name.
      validations (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          that stores validations for max_length, min_length, max_items,
          min_items, exclusive_maximum, inclusive_maximum, exclusive_minimum,
          inclusive_minimum, and regex.
      additional_properties_type (tuple): A tuple of classes accepted
          as additional properties values.
    """
    allowed_values = ...
    validations = ...
    @cached_property
    def additional_properties_type(): # -> tuple[type[bool], type[date], type[datetime], type[dict[Any, Any]], type[float], type[int], type[list[Any]], type[str], type[None]]:
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded
        """
        ...

    _nullable = ...
    @cached_property
    def openapi_types(): # -> dict[str, tuple[type[str]]]:
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded

        Returns
            openapi_types (dict): The key is attribute name
                and the value is attribute type.
        """
        ...

    @cached_property
    def discriminator(): # -> None:
        ...

    attribute_map = ...
    read_only_vars = ...
    _composed_schemas = ...
    required_properties = ...
    @convert_js_args_to_python_args
    def __init__(self, name, source, *args, **kwargs) -> None:
        """CreateCollectionRequest - a model defined in OpenAPI

        Args:
            name (str): The name of the collection to be created. Resource name must be 1-45 characters long, start and end with an alphanumeric character, and consist only of lower case alphanumeric characters or '-'.
            source (str): The name of the index to be used as the source for the collection.

        Keyword Args:
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
        """
        ...

