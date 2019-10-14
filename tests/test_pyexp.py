import pytest

from vc2_bit_widths.pyexp import (
    ensure_py_exp,
    PyExp,
    Constant,
    Argument,
    Subscript,
    BinaryOperator,
)


def test_ensure_py_exp():
    # Passes through PyExp types unchanged
    foo = Argument("foo")
    assert ensure_py_exp(foo) is foo
    
    # Wraps other values in Constant
    assert ensure_py_exp(123) == Constant(123)


class TestConstant(object):
    
    def test_code(self):
        c = Constant("foo")
        assert list(c.get_dependencies()) == []
        assert list(c.get_argument_names()) == []
        assert c.get_definitions() == ""
        assert c.get_expression() == "'foo'"
    
    def test_hash(self):
        assert hash(Constant(123)) == hash(Constant(123))
        assert hash(Constant(123)) != hash(Constant(321))
    
    def test_eq(self):
        assert Constant(123) == Constant(123)
        assert Constant(123) != Constant(321)
        assert Constant(123) != Argument("foo")
    
    def test_repr(self):
        assert repr(Constant("foo")) == "Constant('foo')"
    
    def test_subs(self):
        c = Constant(123)
        assert c.subs({Constant(123): Constant(321)}) == Constant(321)
        assert c.subs({Constant(321): Constant(0)}) is c


class TestArgument(object):
    
    def test_code(self):
        a = Argument("foo")
        assert list(a.get_dependencies()) == []
        assert list(a.get_argument_names()) == ["foo"]
        assert a.get_definitions() == ""
        assert a.get_expression() == "foo"
    
    def test_hash(self):
        assert hash(Argument("foo")) == hash(Argument("foo"))
        assert hash(Argument("foo")) != hash(Argument("bar"))
    
    def test_eq(self):
        assert Argument("foo") == Argument("foo")
        assert Argument("foo") != Argument("bar")
        assert Argument("foo") != Constant("foo")
    
    def test_repr(self):
        assert repr(Argument("foo")) == "Argument('foo')"
    
    def test_subs(self):
        a = Argument("foo")
        assert a.subs({Argument("foo"): Argument("bar")}) == Argument("bar")
        assert a.subs({Argument("bar"): Argument("xxx")}) is a


class TestSubscript(object):
    
    @pytest.fixture
    def exp(self):
        return Argument("foo")
    
    @pytest.fixture
    def key(self):
        return Constant(123)
    
    def test_code(self, exp, key):
        s = Subscript(exp, key)
        assert set(s.get_dependencies()) == set([exp, key])
        assert list(s.get_argument_names()) == []
        assert s.get_definitions() == ""
        assert s.get_expression() == "foo[123]"
    
    def test_hash(self, exp, key):
        assert hash(Subscript(exp, key)) == hash(Subscript(exp, key))
        assert hash(Subscript(exp, key)) != hash(Subscript(Argument("bar"), key))
        assert hash(Subscript(exp, key)) != hash(Subscript(exp, Constant(321)))
    
    def test_eq(self, exp, key):
        assert Subscript(exp, key) == Subscript(exp, key)
        assert Subscript(exp, key) != Subscript(Argument("bar"), key)
        assert Subscript(exp, key) != Subscript(exp, Constant(321))
    
    def test_repr(self, exp, key):
        assert repr(Subscript(exp, key)) == "Subscript(Argument('foo'), Constant(123))"
    
    def test_subs(self, exp, key):
        s = Subscript(exp, key)
        
        # Substitute whole subscript
        assert s.subs({exp[key]: Argument("bar")}) == Argument("bar")
        
        # Substitute expression
        assert s.subs({exp: Argument("bar")}) == Argument("bar")[123]
        
        # Substitute key
        assert s.subs({key: Constant(321)}) == exp[321]
        
        # Substitute nothing relevant
        assert s.subs({Argument("bar"): Argument("xxx")}) is s


class TestBinaryOperator(object):
    
    @pytest.fixture
    def lhs(self):
        return Argument("lhs")
    
    @pytest.fixture
    def rhs(self):
        return Argument("rhs")
    
    def test_code(self, lhs, rhs):
        b = BinaryOperator(lhs, rhs, "+")
        
        b.set_inline(False)
        assert set(b.get_dependencies()) == set([lhs, rhs])
        assert list(b.get_argument_names()) == []
        assert b.get_definitions() == "_{} = lhs + rhs\n".format(id(b))
        assert b.get_expression() == "_{}".format(id(b))
        
        b.set_inline(True)
        assert set(b.get_dependencies()) == set([lhs, rhs])
        assert list(b.get_argument_names()) == []
        assert b.get_definitions() == "".format(id(b))
        assert b.get_expression() == "(lhs + rhs)"
    
    def test_repr(self, lhs, rhs):
        assert repr(BinaryOperator(lhs, rhs, "<<")) == "BinaryOperator(Argument('lhs'), Argument('rhs'), '<<')"
    
    def test_subs(self, lhs, rhs):
        b = lhs + rhs
        
        # Substitute whole subscript: NB: same object must be used in this case
        assert b.subs({b: Argument("bar")}) == Argument("bar")
        
        # NB: in the following we must comprae string for true equality since
        # BinaryOperators are compared by identity
        
        # Substitute LHS 
        assert repr(b.subs({lhs: Argument("foo")})) == repr(Argument("foo") + rhs)
        
        # Substitute RHS
        assert str(b.subs({rhs: Argument("foo")})) == str(lhs + Argument("foo"))
        
        # Substitution which makes operation a no-op should not return a
        # BinaryOperator
        assert b.subs({rhs: Constant(0)}) == lhs
        
        # Substitute nothing relevant
        assert b.subs({Argument("bar"): Argument("xxx")}) is b
    
    @pytest.mark.parametrize("lhs,rhs,operator,expected", [
        # Non no-op arguments
        (Constant(123), Constant(321), "+", None),
        (Constant(123), Constant(321), "-", None),
        (Constant(123), Constant(321), "*", None),
        (Constant(123), Constant(321), "<<", None),
        (Constant(123), Constant(321), ">>", None),
        (Argument("foo"), Argument("foo"), "+", None),
        (Argument("foo"), Argument("foo"), "-", None),
        (Argument("foo"), Argument("foo"), "*", None),
        (Argument("foo"), Argument("foo"), "<<", None),
        (Argument("foo"), Argument("foo"), ">>", None),
        # No-op addition
        (Argument("foo"), Constant(0), "+", Argument("foo")),
        (Constant(0), Argument("foo"), "+", Argument("foo")),
        # No-op subtraction
        (Argument("foo"), Constant(0), "-", Argument("foo")),
        # *Not* no-op subtraction
        (Constant(0), Argument("foo"), "-", None),
        # No-op multiplication
        (Argument("foo"), Constant(1), "*", Argument("foo")),
        (Constant(1), Argument("foo"), "*", Argument("foo")),
        (Argument("foo"), Constant(0), "*", Constant(0)),
        (Constant(0), Argument("foo"), "*", Constant(0)),
        # No-op shift
        (Argument("foo"), Constant(0), "<<", Argument("foo")),
        (Argument("foo"), Constant(0), ">>", Argument("foo")),
        # *Not* no-op shift!
        (Constant(0), Argument("foo"), "<<", None),
        (Constant(0), Argument("foo"), ">>", None),
    ])
    def test_is_no_op(self, lhs, rhs, operator, expected):
        assert BinaryOperator.is_no_op(lhs, rhs, operator) == expected


class TestOperators(object):
    
    def test_binary_operator_casts_to_constant(self):
        exp = PyExp._binary_operator(123, 321, "+")
        
        assert isinstance(exp, BinaryOperator)
        assert exp._lhs == Constant(123)
        assert exp._rhs == Constant(321)
        assert exp._operator == "+"
    
    def test_binary_operator_uses_pyexp_passed_in(self):
        lhs = Argument("lhs")
        rhs = Argument("rhs")
        exp = PyExp._binary_operator(lhs, rhs, "*")
        
        assert isinstance(exp, BinaryOperator)
        assert exp._lhs == lhs
        assert exp._rhs == rhs
        assert exp._operator == "*"
    
    def test_binary_operator_checks_for_passthrough(self):
        lhs = Argument("lhs")
        assert PyExp._binary_operator(lhs, 1, "*") is lhs
    
    @pytest.mark.parametrize("operator", ["+", "-", "*", "/", "//", "<<", ">>"])
    def test_binary_operators(self, operator):
        a = Argument("a")
        b = Argument("b")
        
        # Native operation
        exp = eval("a {} b".format(operator), {}, {"a": a, "b": b})
        assert isinstance(exp, BinaryOperator)
        assert exp._lhs == a
        assert exp._rhs == b
        assert exp._operator == operator
        
        # RHS is not PyExp
        exp = eval("a {} 123".format(operator), {}, {"a": a})
        assert isinstance(exp, BinaryOperator)
        assert exp._lhs == a
        assert exp._rhs == Constant(123)
        assert exp._operator == operator
        
        # Reverse-operator methods (and LHS is not PyExp)
        exp = eval("123 {} a".format(operator), {}, {"a": a})
        assert isinstance(exp, BinaryOperator)
        assert exp._lhs == Constant(123)
        assert exp._rhs == a
        assert exp._operator == operator
    
    def test_subscript(self):
        a = Argument("a")
        c = Constant(123)
        
        s = a[c]
        
        assert isinstance(s, Subscript)
        assert s.exp is a
        assert s.key is c


class TestAutoInline(object):
    
    def test_only_used_once(self):
        a = Argument("a")
        b = Argument("b")
        c = Argument("c")
        
        a_plus_b = a + b
        a_plus_b_plus_c = a_plus_b + c
        
        a_plus_b_plus_c._auto_inline()
        
        assert a_plus_b_plus_c._inline is True
        assert a_plus_b._inline is True
    
    def test_used_multiple_times(self):
        a = Argument("a")
        b = Argument("b")
        c = Argument("c")
        
        a_plus_b = a + b
        twice_a_plus_b = a_plus_b + a_plus_b
        
        twice_a_plus_b._auto_inline()
        
        assert twice_a_plus_b._inline is True
        assert a_plus_b._inline is False
    
    def test_nested_out_of_lineable_values(self):
        a = Argument("a")
        b = a + 1  # Depth 2 -> 0 (de-inlined)
        c = b + 1  # Depth 1
        d = c + 1  # Depth 2 -> 0 (de-inlined)
        e = d + 1  # Depth 1
        f = e + 1  # Depth 0
        
        f._auto_inline(2)
        assert not b._inline
        assert c._inline
        assert not d._inline
        assert e._inline
        assert f._inline


class TestGetAllArgumentNames(object):
    
    def test_just_argument(self):
        a = Argument("a")
        assert a.get_all_argument_names() == ["a"]
    
    def test_no_repeats(self):
        a = Argument("a")
        b = Argument("b")
        
        exp = a + b - a
        
        assert exp.get_all_argument_names() == ["a", "b"]
    
    def test_ordered(self):
        a = Argument("a")
        b = Argument("b")
        
        exp = a + b
        assert exp.get_all_argument_names() == ["a", "b"]
        
        exp = b + a
        assert exp.get_all_argument_names() == ["a", "b"]


class TestGenerateCode(object):
    
    def test_arguments(self):
        assert Argument("foo").generate_code() == (
            "def f(foo):\n"
            "    return foo\n"
        )
        
        # Arguments should be in alphabetical order
        foo_bar = Argument("foo") + Argument("bar")
        assert foo_bar.generate_code() == (
            "def f(bar, foo):\n"
            "    return (foo + bar)\n"
        )
    
    def test_dependencies(self):
        ten = Constant(10)
        twice_ten = ten + ten
        twice_ten_squared = twice_ten * twice_ten
        one = twice_ten_squared // twice_ten_squared
        
        assert one.generate_code() == (
            "def f():\n"
            "    _{tt} = 10 + 10\n"
            "    _{tts} = _{tt} * _{tt}\n"
            "    return (_{tts} // _{tts})\n"
        ).format(
            tt=id(twice_ten),
            tts=id(twice_ten_squared),
        )

class TestSubs(object):
    
    # Specific tests of global operation of the subs method
    
    def test_subs_in_reused_binary_operator(self):
        # Check if operands to a re-used binary operator are changed, the new
        # expression still re-uses the updated binary operator.
        a = Argument("a")
        b = Argument("b")
        c = Argument("c")
        d = Argument("d")
        
        a_plus_b = a + b
        
        a_plus_b_squared = a_plus_b * a_plus_b
        
        # Sanity check
        assert a_plus_b_squared.lhs is a_plus_b_squared.rhs
        
        c_plus_d_squared = a_plus_b_squared.subs({a: c, b: d})
        assert c_plus_d_squared.lhs is c_plus_d_squared.rhs
    
    def test_swap(self):
        # Check variable swapping is supported (i.e. items in before and after
        # are same)
        
        a = Argument("a")
        b = Argument("b")
        
        a_sub_b = a - b
        
        b_sub_a = a_sub_b.subs({a: b, b: a})
        
        assert str(b_sub_a) == str(b - a)


def test_make_function():
    a = Argument("a")
    b = Argument("b")
    c = Argument("c")
    
    abc = (a + b) * c
    
    f = abc.make_function()
    
    assert f(a=1, b=2, c=10) == (1 + 2) * 10


def test_deeply_nested_expression():
    a = Argument("a")
    nest_depth = 1000
    for _ in range(nest_depth):
        a += 1
    
    f = a.make_function()
    
    assert f(10) == 10 + nest_depth
