- [] Read up on the build tools and what not for pybind11
- [] Just expose the computationally heavy methods to Python through pybind11. My thinking is: 
    * wrap Parameters in a separate class and expose so that a convenient constructor can be provided from some python object like a dictionary or namedtuple.
    * wrap GW_model so that parameters can be set with the PyParameters object, and only expose the computationally heavy methods.
- [] Wrap the exported class in a separate class within Python for convenience, e.g. calculating currents with numpy arrays, serializing results, plotting etc.