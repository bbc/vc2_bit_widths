r"""
:py:mod:`vc2_bit_widths.pyexp`: Construct Python programs implementing arithmetic expressions
=============================================================================================

This module provides the :py:class:`PyExp` class which tracks arithmetic
operations performed on its instances. These objects may later be used to
generate a Python function which implements these steps.


Motivation
----------

When constructing test patterns it is often necessary to compute the output of
a single filter output. Typically filters are described in terms of lifting
steps which compute the filter output for an entire picture at once, even when
only a single pixel or intermediate value is required. Using :py:class:`PyExp`,
it is possible to automatically extract a function which implements just the
computations required to produce a single filter output. In this way,
individual filter outputs may be efficiently computed in isolation.


Getting started
---------------

The :py:class:`pyexp` module provides a series of subclasses of the abstract
:py:class:`PyExp` base class. Together these subclasses may be used to define
Python expressions which may be compiled into a function.

For example, using :py:class:`Constant` we can define a pair of constants and
add them together::

    >>> from vc2_bit_widths.pyexp import Constant
    
    >>> five = Constant(5)
    >>> nine = Constant(9)
    >>> five_add_nine = five + nine
    
    >>> print(five_add_nine)
    BinaryOperator(Constant(5), Constant(9), '+')

Rather than immediately computing a result, adding the :py:class:`Constant`\ s
together produced a new :py:class:`PyExp` (of type :py:class:`BinaryOperator`)
representing the computation to be carried out. Using the
:py:meth:`~PyExp.make_function` method, we can create a Python function which
actually evaluates the expression::

    >>> compute_five_add_nine = five_add_nine.make_function()
    >>> compute_five_add_nine()
    14

The :py:meth:`~PyExp.generate_code` method may be used to inspect the generated
code.  Here we can see that the generated function does actually perform the
computation on demand::

    >>> print(five_add_nine.generate_code())
    def f():
        return (5 + 9)

To build more interesting functions we can define :py:class:`Argument`\ s.  The
names passed to the :py:class:`Argument` constructor become the argument names
in the generated function::

    >>> from vc2_bit_widths.pyexp import Argument
    
    >>> a = Argument("a")
    >>> b = Argument("b")
    >>> average_expr = (a + b) / Constant(2.0)
    >>> average = average_expr.make_function()
    
    >>> average(a=10, b=15)
    12.5

:py:class:`PyExp` instances support many Python operators and will
automatically wrap non-:py:class:`PyExp` values in :py:class:`Constant`. For
example::

    >>> array = Argument("array")
    >>> average_of_first_and_last_expr = (array[0] + array[-1]) / 2.0
    >>> average_of_first_and_last = average_of_first_and_last_expr.make_function()
    
    >>> average_of_first_and_last([3, 4, 5, 6])
    4.5


Carving functions out of existing code
--------------------------------------

Because :py:class:`PyExp` objects support the usual Python operators they can
be passed as arguments to existing code. For example, consider the following
function designed to operate on arrays of numbers::

    >>> def multiply_by_place(array):
    ...     return [value * i for i, value in enumerate(array)]
    
    >>> # Demo
    >>> multiply_by_place([4, 3, 5, 11])
    [0, 3, 10, 33]

Now, instead of passing in an array of numbers, lets pass in an array of
:py:class:`PyExp` values::

    >>> values = Argument("values")
    >>> out = multiply_by_place([values[i] for i in range(4)])
    
    >>> from pprint import pprint
    >>> pprint(out)
    [Constant(0),
     Subscript(Argument('values'), Constant(1)),
     BinaryOperator(Subscript(Argument('values'), Constant(2)), Constant(2), '*'),
     BinaryOperator(Subscript(Argument('values'), Constant(3)), Constant(3), '*')]

Our function now returns a list of :py:class:`PyExp` values. Using these we can
generate functions which just compute one of the output values of
``multiply_by_place`` in isolation::

    >>> compute_third_output = out[2].make_function()
    >>> compute_third_output([4, 3, 5, 11])
    10

We can verify that the generated function is genuinely computing only the
required value (and not just computing everything and throwing most of it
away) by inspecting its source code::

    >>> print(out[2].generate_code())
    def f(values):
        return (values[2] * 2)


Manipulating expressions
------------------------

:py:class:`PyExp` supports simple manipulation of expressions in the form of
the :py:meth:`~PyExp.subs` method which substitutes subexpressions within an
expression. For example::

    >>> a = Argument("a")
    >>> b = Argument("b")
    >>> c = Argument("c")
    
    >>> exp = a["deeply"]["nested"]["subscript"] + b
    BinaryOperator(Subscript(Subscript(Subscript(Argument('a'), Constant('deeply')), Constant('nested')), Constant('subscript')), Argument('b'), '+')
    
    >>> exp2 = exp.subs({a["deeply"]["nested"]["subscript"]: c})
    >>> print(exp2)
    BinaryOperator(Argument('c'), Argument("b"), "+")

This functionality is intended to allow simple optimisations such as constant
folding (substituting constants in place of variables) and replacing
subscripted values with :py:class:`Argument`\ s to reduce overheads.


API
---

.. autoclass:: PyExp
    :exclude-members: set_inline, get_dependencies, get_argument_names, get_definitions, get_expression

.. autoclass:: Constant
    :members: value
    :exclude-members: get_dependencies, get_argument_names, get_definitions, get_expression

.. autoclass:: Argument
    :members: name
    :exclude-members: get_dependencies, get_argument_names, get_definitions, get_expression

.. autoclass:: Subscript
    :members: exp, key
    :exclude-members: get_dependencies, get_argument_names, get_definitions, get_expression, subs

.. autoclass:: BinaryOperator
    :members: lhs, rhs, operator
    :exclude-members: set_inline, get_dependencies, get_argument_names, get_definitions, get_expression, subs, is_no_op

"""


def ensure_py_exp(maybe_exp):
    """
    If the passed value is a :py:class:`PyExp`, return it unchanged. Otherwise,
    return a :py:class:`Constant` containing the passed value.
    """
    if isinstance(maybe_exp, PyExp):
        return maybe_exp
    else:
        return Constant(maybe_exp)


class PyExp(object):
    """
    Abstract Base class for Python Expression objects.
    """
    
    # Implementers must implement the following methods:
    # 
    # * get_dependencies
    # * get_argument_names
    # * get_definitions
    # * get_expression
    #
    # Implementers may also need to implement:
    # * subs
    
    def __init__(self):
        pass
    
    def set_inline(self, inline):
        """
        Hint whether this expression should be inlined or not. Returns True if
        the hint will be obeyed, False otherwise.
        """
        return False
    
    def get_dependencies(self):
        r"""
        Return an iterable over the :py:class:`PyExp`\ s which this :py;class:`PyExp`
        directly depends on to compute. (Does not recurse.)
        """
        raise NotImplementedError()
    
    def get_argument_names(self):
        """
        Return an iterable of Python argument names which this expression
        directly depends on (Does not recurse).
        """
    
    def get_definitions(self):
        """
        Return a string containing Python code which defines any variables
        required by the :py:attr:`expression`. (Does not include recursive
        definitions for any dependencies!)
        """
        raise NotImplementedError()
    
    def get_expression(self):
        """
        Return a string containing a Python expression evaluating to this value.
        """
        raise NotImplementedError()
    
    def subs(self, substitutions):
        """
        Substitute subexpressions. Returns a new :py:class:`PyExp` with the
        substitutions applied.
        
        Parameters
        ==========
        substitutions : {before: after, ...}
            A dictionary mapping from :py:class:`PyExp` to :py:class:`PyExp`.
            
            This dictionary may be mutated during calls to subs (with
            additional entries being inserted).
            
            .. warning::
                
                Unlike other :py:class:`PyExp` types,
                :py:class:`BinaryOperator` instances are compared by identity
                and so using these as 'before' values may not produce the
                expected results, e.g.::
                
                    >>> a = Argument(a)
                    >>> b = Argument(b)
                    
                    >>> a == a
                    True
                    >>> b[123] == b[123]
                    True
                    >>> (a + b) == (a + b)
                    False
                    >>> a_plus_b = a + b
                    >>> a_plus_b == a_plus_b
                    True
        
        Returns
        =======
        exp : :py:class:`PyExp`
            A new :py:class:`PyExp` with the specified substitutions enacted.
        """
        # This base implementation should be extended by subclasses which
        # depend on other PyExps.
        substitution = substitutions.get(self)
        if substitution is not None:
            return substitution
        else:
            return self
    
    def _auto_inline(self, max_depth=64):
        """
        Recursively inline all dependencies whose values are only used once and
        values which are used in sufficiently deeply nested expressions as to
        cause a stack overflow in Python's parser.
        """
        visited = set()
        exp_depths = {}
        
        to_visit = [self]
        
        while to_visit:
            exp = to_visit.pop(-1)
            depth = exp_depths.setdefault(exp, 0)
            if exp in visited:
                # Used in multiple places, don't inline
                if exp.set_inline(False):
                    # Only reset depth if inline hint is obeyed
                    exp_depths[exp] = 0
            else:
                # Initially assume used only once and inline unless depth is
                # very deep
                if depth < max_depth:
                    exp.set_inline(True)
                else:
                    if exp.set_inline(False):
                        depth = 0
                
                visited.add(exp)
                for dependant in exp.get_dependencies():
                    to_visit.append(dependant)
                    if depth + 1 > exp_depths.get(dependant, 0):
                        exp_depths[dependant] = depth + 1
    
    def get_all_argument_names(self):
        """
        Returns the list of argument names in the order they are used by the
        generated function.
        """
        visited = set()
        argument_names = set()
        
        to_visit = [self]
        while to_visit:
            exp = to_visit.pop(-1)
            if exp not in visited:
                visited.add(exp)
                argument_names.update(exp.get_argument_names())
                for dep in exp.get_dependencies():
                    to_visit.append(dep)
        
        return sorted(argument_names)
    
    def generate_code(self, function_name="f"):
        """
        Generate a Python function definition string which, once compiled, can
        be used to evaluate the expression embodied by this :py:class:`PyExp`.
        
        See also :py:meth:`make_function`.
        
        Parameters
        ==========
        function_name : str
            The name to give to the generated function
        """
        self._auto_inline()
        
        # The PyExps visited so far
        visited = set()
        
        # The complete set of statements to be execute, in the order required
        # to generate the final output.
        statements = []
        
        # A stack of (action, argument) pairs which will be used to iterate
        # over this expression and dependencies in execution order.
        #
        # Two types of action are defined:
        # * "emit": Add a value to be added to the statements list
        # * "visit": Visit a particular expression emitting any dependencies'
        #   definitions and the expression's definition.
        stack = [
            ("emit", "return {}".format(self.get_expression())),
            ("visit", self),
        ]
        while stack:
            action, argument = stack.pop(-1)
            if action == "visit":
                if argument not in visited:
                    visited.add(argument)
                    stack.append(("emit", argument.get_definitions()))
                    for dep in argument.get_dependencies():
                        stack.append(("visit", dep))
            elif action == "emit":
                statements.append(argument)
        
        return (
            "def {}({}):\n"
            "    {}\n"
        ).format(
            function_name,
            ", ".join(self.get_all_argument_names()),
            "".join(statements).replace("\n", "\n    "),
        )
    
    def make_function(self, function_name="f"):
        r"""
        Create a Python function which implements the expression embodied by
        this :py:class:`PyExp`.
        
        The returned function will expect (as keyword arguments) any
        :py:class:`Argument`\ s used in the definition of this
        :py:class:`PyExp`.
        
        See also :py:meth:`generate_code`.
        """
        loc = {}
        exec(self.generate_code(function_name), {}, loc)
        return loc[function_name]
    
    def __hash__(self):
        return hash(id(self))
    
    def __eq__(self, other):
        return self is other
    
    @staticmethod
    def _binary_operator(lhs, rhs, operator):
        lhs = ensure_py_exp(lhs)
        rhs = ensure_py_exp(rhs)

        no_op_passthrough = BinaryOperator.is_no_op(lhs, rhs, operator)
        if no_op_passthrough:
            return no_op_passthrough
        else:
            return BinaryOperator(lhs, rhs, operator)
    
    # Arithmetic operators
    
    def __add__(self, other):
        return self._binary_operator(self, other, "+")
    def __radd__(self, other):
        return self._binary_operator(other, self, "+")
    
    def __sub__(self, other):
        return self._binary_operator(self, other, "-")
    def __rsub__(self, other):
        return self._binary_operator(other, self, "-")
    
    def __mul__(self, other):
        return self._binary_operator(self, other, "*")
    def __rmul__(self, other):
        return self._binary_operator(other, self, "*")
    
    def __floordiv__(self, other):
        return self._binary_operator(self, other, "//")
    def __rfloordiv__(self, other):
        return self._binary_operator(other, self, "//")
    
    # Python 3.x
    def __truediv__(self, other):
        return self._binary_operator(self, other, "/")
    def __rtruediv__(self, other):
        return self._binary_operator(other, self, "/")
    
    # Python 2.x
    def __div__(self, other):
        return self._binary_operator(self, other, "/")
    def __rdiv__(self, other):
        return self._binary_operator(other, self, "/")
    
    def __lshift__(self, other):
        return self._binary_operator(self, other, "<<")
    def __rlshift__(self, other):
        return self._binary_operator(other, self, "<<")
    
    def __rshift__(self, other):
        return self._binary_operator(self, other, ">>")
    def __rrshift__(self, other):
        return self._binary_operator(other, self, ">>")
    
    # Subscript
    
    def __getitem__(self, key):
        key = ensure_py_exp(key)
        return Subscript(self, key)


class Constant(PyExp):
    """
    A constant value.
    
    Parameters
    ==========
    value : any
        Must be serialised to a valid Python expression by :py:func:`repr`
    """
    
    def __init__(self, value):
        super(Constant, self).__init__()
        
        self._value = value
    
    @property
    def value(self):
        """The value of this constant."""
        return self._value
    
    def get_dependencies(self):
        return []
    
    def get_argument_names(self):
        return []
    
    def get_definitions(self):
        return ""
    
    def get_expression(self):
        return repr(self._value)
    
    def __hash__(self):
        return hash(self._value)
    
    def __eq__(self, other):
        return type(self) == type(other) and self._value == other._value
    
    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.value)


class Argument(PyExp):
    """
    An named argument which will be expected by the Python function
    implementing this expression.
    
    Parameters
    ==========
    name : str
        A valid Python argument name
    """
    
    def __init__(self, name):
        super(Argument, self).__init__()
        
        self._name = name
    
    @property
    def name(self):
        """The name of this argument."""
        return self._name
    
    def get_dependencies(self):
        return []
    
    def get_argument_names(self):
        return [self._name]
    
    def get_definitions(self):
        return ""
    
    def get_expression(self):
        return self._name
    
    def __hash__(self):
        return hash(self._name)
    
    def __eq__(self, other):
        return type(self) == type(other) and self._name == other._name
    
    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.name)


class Subscript(PyExp):
    """
    Subscript an expression, i.e. ``exp[key]``.
    
    Parameters
    ==========
    exp : :py:class:`PyExp`
        The value to be subscripted.
    key : :py:class:`PyExp`
        The key to access.
    """
    
    def __init__(self, exp, key):
        super(Subscript, self).__init__()
        
        self._exp = exp
        self._key = key
    
    @property
    def exp(self):
        """The expression being subscripted."""
        return self._exp
    
    @property
    def key(self):
        """The subscript key."""
        return self._key
    
    def get_dependencies(self):
        return [self._exp, self._key]
    
    def get_argument_names(self):
        return []
    
    def get_definitions(self):
        return ""
    
    def get_expression(self):
        return "{}[{}]".format(
            self._exp.get_expression(),
            self._key.get_expression(),
        )
    
    def subs(self, substitutions):
        substitution = super(Subscript, self).subs(substitutions)
        if substitution is self:
            # May need to make substitution in dependency
            new_exp = self._exp.subs(substitutions)
            new_key = self._key.subs(substitutions)
            if new_exp is not self._exp or new_key is not self._key:
                substitution = Subscript(new_exp, new_key)
            
            return substitution
        else:
            return substitution
    
    def __hash__(self):
        return hash((self._exp, self._key))
    
    def __eq__(self, other):
        return (
            type(self) == type(other) and
            self._exp == other._exp and
            self._key == other._key
        )
    
    def __repr__(self):
        return "{}({!r}, {!r})".format(type(self).__name__, self._exp, self._key)


class BinaryOperator(PyExp):
    r"""
    A binary operation applied to two :py:class:`PyExp`\ s, e.g. addition.
    
    Parameters
    ==========
    lhs : :py:class:`PyExp`
    rhs : :py:class:`PyExp`
        The arguments to the operator.
    operator : str
        The python operator symbol to apply (e.g. "+").
    """
    
    @staticmethod
    def is_no_op(lhs, rhs, operator):
        """
        If the proposed binary operation would be a equivalent to a
        no-operation, returns the :py:class:`PyExp` for the argument which
        would be the result. Returns None otherwise.
        
        This function is not necessarily complete (e.g. it may return None in
        cases where the operation is a no-op) and should be used as an
        optimisation hint only.
        """
        if operator == "*":
            if isinstance(lhs, Constant) and lhs.value == 1:
                return rhs
            if isinstance(rhs, Constant) and rhs.value == 1:
                return lhs
            if isinstance(lhs, Constant) and lhs.value == 0:
                return lhs
            if isinstance(rhs, Constant) and rhs.value == 0:
                return rhs
        elif operator == "+":
            if isinstance(lhs, Constant) and lhs.value == 0:
                return rhs
            if isinstance(rhs, Constant) and rhs.value == 0:
                return lhs
        elif operator == "-":
            if isinstance(rhs, Constant) and rhs.value == 0:
                return lhs
        elif operator == "<<":
            if isinstance(rhs, Constant) and rhs.value == 0:
                return lhs
        elif operator == ">>":
            if isinstance(rhs, Constant) and rhs.value == 0:
                return lhs
        
        return None
    
    def __init__(self, lhs, rhs, operator):
        super(BinaryOperator, self).__init__()
        
        self._lhs = lhs
        self._rhs = rhs
        self._operator = operator
        
        self._inline = False
    
    def set_inline(self, inline):
        self._inline = inline
        return True
    
    @property
    def lhs(self):
        """The expression on the left-hand side of the operator."""
        return self._lhs
    
    @property
    def rhs(self):
        """The expression on the right-hand side of the operator."""
        return self._rhs
    
    @property
    def operator(self):
        """The operator, as a string."""
        return self._operator
    
    def get_dependencies(self):
        return [self._lhs, self._rhs]
    
    def get_argument_names(self):
        return []
    
    def get_definitions(self):
        if self._inline:
            return ""
        else:
            return "_{} = {} {} {}\n".format(
                id(self),
                self._lhs.get_expression(),
                self._operator,
                self._rhs.get_expression(),
            )
    
    def get_expression(self):
        if self._inline:
            return "({} {} {})".format(
                self._lhs.get_expression(),
                self._operator,
                self._rhs.get_expression(),
            )
        else:
            return "_{}".format(id(self))
    
    def subs(self, substitutions):
        substitution = super(BinaryOperator, self).subs(substitutions)
        if substitution is self:
            # May need to make substitution in dependency
            new_lhs = self._lhs.subs(substitutions)
            new_rhs = self._rhs.subs(substitutions)
            if new_lhs is not self._lhs or new_rhs is not self._rhs:
                no_op_passthrough = BinaryOperator.is_no_op(new_lhs, new_rhs, self._operator)
                if no_op_passthrough:
                    substitution = no_op_passthrough
                else:
                    substitution = BinaryOperator(new_lhs, new_rhs, self._operator)
                
                # To ensure equivalent behaviour we must replace all other uses
                # of this BinaryOperator with the new version, otherwise every
                # use of this expression will be substituted with a unique
                # BinaryOperator and the computation will be repeated.  This is
                # an artefact of binary operators comparing by instance
                # identity and not by value.
                substitutions[self] = substitution
            
            return substitution
        else:
            return substitution
    
    def __repr__(self):
        return "{}({!r}, {!r}, {!r})".format(type(self).__name__, self._lhs, self._rhs, self._operator)
