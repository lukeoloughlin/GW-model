def create_pybind11_params_boilerplate(
    model_data: dict,
    class_name: str,
    python_name: str,
    template_parameters: None | str = None,
) -> str:
    """Generate the boilerplate for exported parameters struct."""
    if template_parameters is not None:
        class_name += f"<{template_parameters}>"
    out_str = f'py::class_<{class_name}>(m, "{python_name}")'
    for name, info in model_data["parameters"].items():
        out_str += f"""
        .def_readwrite("{name}", &{class_name}::{name}, "{info['description']}")"""
    return out_str
