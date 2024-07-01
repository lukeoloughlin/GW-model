from typing import Any, List, Callable, Tuple, TypeVar, Sequence
from functools import partial

CXXStruct = TypeVar("CXXStruct")


def __update_from_parameter_dict(self, parameter_dict: dict, valid_fields: List[str]):
    for name, value in parameter_dict.items():
        try:
            assert name in valid_fields
            setattr(self, name, value)
        except Exception as e:
            raise AttributeError(
                f"Object of type {type(self)} has no attribute {name}."
            ) from e


def from_cpp_struct(
    struct: CXXStruct,
    additional_attributes: None | Sequence[str | Tuple[str, Any]] = None,
) -> Callable[[Any], Any]:
    def transform_cls(cls: Any) -> Any:
        def getter(self, field):
            return getattr(self.__cpp_struct, field)

        def setter(self, value, field):
            setattr(self.__cpp_struct, field, value)

        def deleter(self):
            pass

        valid_fields = []
        for field in dir(struct):
            if not (field.startswith("__") and field.endswith("__")):
                valid_fields.append(field)
                get_field = partial(getter, field=field)
                set_field = partial(setter, field=field)
                setattr(
                    cls,
                    field,
                    property(get_field, set_field, deleter, "Parameter: " + field),
                )

        defaults_storage = []
        required = []
        if additional_attributes is not None:
            for attr in additional_attributes:
                if isinstance(attr, tuple):
                    if not isinstance(attr[0], str):
                        raise ValueError(
                            f"Additional attributes must be specified by a string or a (string, default value) tuple. Got {attr}."
                        )
                    defaults_storage.append(attr)
                    valid_fields.append(attr[0])
                else:
                    if not isinstance(attr, str):
                        raise ValueError(
                            f"Additional attributes must be specified by a string or a (string, default value) tuple. Got {attr}."
                        )
                    required.append(attr)
                    valid_fields.append(attr)

        setattr(cls, "cpp_struct", property(lambda self: self.__cpp_struct))

        try:
            # check that the default constructor is defined
            struct()  # type: ignore
        except Exception as e:
            raise AttributeError(f"No default constructor for type {struct}") from e

        def init(self, cpp_struct=None, **parameter_kwargs):
            if len(required) > 0 and parameter_kwargs is None:
                raise AttributeError(
                    f"Must explicity provide initialisation values for {required}"
                )
            if len(required) > 0:
                for name in required:
                    if not (name in parameter_kwargs):
                        raise AttributeError(
                            f"Must provide initialisation value for {name}"
                        )

            self.__cpp_struct = struct() if (cpp_struct is None) else cpp_struct
            for name, value in defaults_storage:
                setattr(self, name, value)

            if parameter_kwargs is not None:
                __update_from_parameter_dict(self, parameter_kwargs, valid_fields)

        cls.__init__ = init

        return cls

    return transform_cls
