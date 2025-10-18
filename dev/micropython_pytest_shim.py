import os
class raises():
    def __init__(self, err):
        self.err = err

    def __enter__(self):
        pass

    def __exit__(self, *args):
        if len(args) < 2 or not isinstance(args[1], self.err):
            raise ValueError("Did not raise")
        return True


def istestfunc(func):
    return (
        hasattr(func, "__call__")
        and getattr(func, "__name__", "<lambda>") != "<lambda>"
    )

# import attr

# @attr.s(frozen=True)
class Mark(object):

    def __init__(self, name, args, kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def combined_with(self, other):
        """
        :param other: the mark to combine with
        :type other: Mark
        :rtype: Mark

        combines by appending args and merging the mappings
        """
        assert self.name == other.name
        return Mark(
            self.name, self.args + other.args, dict(self.kwargs, **other.kwargs)
        )


# @attr.s
class MarkDecorator(object):
#     mark = attr.ib(validator=attr.validators.instance_of(Mark))

#     name = alias("mark.name")
#     args = alias("mark.args")
#     kwargs = alias("mark.kwargs")
    def __init__(self, mark):
        self.mark = mark
        self.name = mark.name
        self.args = mark.args
        self.kwargs = mark.kwargs

    @property
    def markname(self):
        return self.name  # for backward-compat (2.4.1 had this attr)

    def __eq__(self, other):
        return self.mark == other.mark if isinstance(other, MarkDecorator) else False

    def __repr__(self):
        return "<MarkDecorator %r>" % (self.mark,)

    def with_args(self, *args, **kwargs):
        """ return a MarkDecorator with extra arguments added

        unlike call this can be used even if the sole argument is a callable/class

        :return: MarkDecorator
        """

        mark = Mark(self.name, args, kwargs)
        return self.__class__(self.mark.combined_with(mark))

    def __call__(self, *args, **kwargs):
        """ if passed a single callable argument: decorate it with mark info.
            otherwise add *args/**kwargs in-place to mark information. """
        if args and not kwargs:
            func = args[0]
            is_class = False
            if len(args) == 1 and (istestfunc(func) or is_class):
                return func
        return self.with_args(*args, **kwargs)

class MarkGenerator(object):
    _config = None
    _markers = set()

    def __getattr__(self, name):
        if name[0] == "_":
            raise AttributeError("Marker name must NOT start with underscore")
        return MarkDecorator(Mark(name, (), {}))

mark = MarkGenerator()
