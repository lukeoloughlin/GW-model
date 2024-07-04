def create_parameters_boilerplate(model_data: dict) -> str:
    """Generate the boilerplate for parameters class fields."""
    out_str = ""
    for name, info in model_data["parameters"].items():
        if info["constraint"] == "positive":
            constraint_fn = "assert_positive"
            constraint_symbol = "> 0"
        elif info["constraint"] == "non-negative":
            constraint_fn = "assert_nonnegative"
            constraint_symbol = ">= 0"
        out_str += f"""
    @property
    def {name}(self) -> float:
        '''float: {info["description"]}. Value must be {constraint_symbol}. Defaults to {info["default"]}.
        '''
        return self.__cxx_struct.{name}

    @{name}.setter
    def {name}(self, value: float) -> None:
        {constraint_fn}(value, "{name}")
        self.__cxx_struct.{name} = value
    """
    return out_str
