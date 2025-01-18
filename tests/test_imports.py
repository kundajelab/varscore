def test_varscore_exposes_add_n_closest_elements():
    import varscore

    # Check if the function exists in the package namespace
    assert hasattr(varscore, "add_n_closest_elements"), "add_n_closest_elements is not exposed in varscore"
    
    # Optionally, check the function's type
    from types import FunctionType
    assert isinstance(varscore.add_n_closest_elements, FunctionType), "add_n_closest_elements is not a function"
