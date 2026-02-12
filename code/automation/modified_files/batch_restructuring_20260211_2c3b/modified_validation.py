"""Helper functions to validate input data and produce error messages."""
from __future__ import print_function, division, absolute_import

import imgaug as ia


def convert_iterable_to_string_of_types(iterable):
    """Convert an iterable of values to a string of their types.

    Parameters
    ----------
    iterable : iterable
        An iterable of variables, e.g. a list of integers.

    Returns
    -------
    str
        String representation of the types in `iterable`. One per item
        in `iterable`. Separated by commas.

    """
    types = [str(type(item)) for item in iterable]
    return ", ".join(types)


def is_iterable_of(iterable, expected_types):
    """Check whether `iterable` contains only instances of given classes.

    Parameters
    ----------
    iterable : iterable
        An iterable of items that will be matched against `expected_types`.

    expected_types : type or iterable of type
        One or more classes that each item in `iterable` must be an instanceof.
        If this is an iterable, a single match per item is enough.

    Returns
    -------
    bool
        Whether `iterable` only contains instances of `expected_types`.
        If `iterable` was empty, ``True`` will be returned.

    """
    if not ia.is_iterable(iterable):
        return False

    for item in iterable:
        if not isinstance(item, expected_types):
            return False

    return True


def assert_is_iterable_of(iterable, expected_types):
    """Assert that `iterable` only contains instances of given classes.

    Parameters
    ----------
    iterable : iterable
        See :func:`~imgaug.validation.is_iterable_of`.

    expected_types : type or iterable of type
        See :func:`~imgaug.validation.is_iterable_of`.

    """
    valid = is_iterable_of(iterable, expected_types)
    if not valid:
        expected_types_str = (
            ", ".join([t.__name__ for t in expected_types])
            if not isinstance(expected_types, type)
            else expected_types.__name__)
        if not ia.is_iterable(iterable):
            raise AssertionError(
                "Expected an iterable of the following types: %s. "
                "Got instead a single instance of: %s." % (
                    expected_types_str,
                    type(iterable).__name__)
            )

        raise AssertionError(
            "Expected an iterable of the following types: %s. "
            "Got an iterable of types: %s." % (
                expected_types_str,
                convert_iterable_to_string_of_types(iterable))
        )