if validate < 1:
    raise ValueError("validate is an integer, it must have a value >=1, "
                    "got %d instead." % (validate,))