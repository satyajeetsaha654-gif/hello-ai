# core/services/math_service.py

from __future__ import annotations

import ast
import math
import operator
from typing import Union


Number = Union[int, float]


# ============================================================
# LIMITS
# ============================================================

MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 100
MAX_ABS_VALUE = 1e100
MAX_EXPONENT = 100


# ============================================================
# ALLOWED OPERATORS
# ============================================================

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# ============================================================
# VALIDATION
# ============================================================

def _is_valid_number(value: object) -> bool:
    """
    শুধুমাত্র int/float গ্রহণ করে।
    bool-কে number হিসেবে গ্রহণ করা হবে না।
    """

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    if isinstance(value, float) and not math.isfinite(value):
        return False

    return True


def _check_value(value: Number) -> Number:
    """
    Calculation result-এর size এবং validity check করে।
    """

    if not _is_valid_number(value):
        raise ValueError("Invalid numeric value")

    if abs(value) > MAX_ABS_VALUE:
        raise ValueError("Result is too large")

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result is not finite")

    return value


def _count_nodes(tree: ast.AST) -> int:
    count = 0

    for _ in ast.walk(tree):
        count += 1

        if count > MAX_AST_NODES:
            raise ValueError("Expression is too complex")

    return count


# ============================================================
# SAFE EVALUATOR
# ============================================================

def _evaluate(node: ast.AST) -> Number:
    """
    AST node safely evaluate করে।

    eval() ব্যবহার করা হচ্ছে না।
    """

    # --------------------------------------------------------
    # NUMBER
    # --------------------------------------------------------

    if isinstance(node, ast.Constant):
        value = node.value

        if not _is_valid_number(value):
            raise ValueError("Only numeric constants are allowed")

        return _check_value(value)

    # --------------------------------------------------------
    # UNARY + / -
    # --------------------------------------------------------

    if isinstance(node, ast.UnaryOp):
        operator_function = _UNARY_OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("Unsupported unary operator")

        operand = _evaluate(node.operand)

        try:
            result = operator_function(operand)
        except Exception as exc:
            raise ValueError("Invalid unary operation") from exc

        return _check_value(result)

    # --------------------------------------------------------
    # BINARY OPERATIONS
    # --------------------------------------------------------

    if isinstance(node, ast.BinOp):
        operator_function = _BINARY_OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("Unsupported operator")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        # ----------------------------------------------------
        # POWER LIMIT
        # ----------------------------------------------------

        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError("Exponent is too large")

            if right > 0 and abs(left) > 1 and right > MAX_EXPONENT:
                raise ValueError("Power operation is too large")

        # ----------------------------------------------------
        # DIVISION / FLOOR DIVISION / MODULO
        # ----------------------------------------------------

        if isinstance(
            node.op,
            (ast.Div, ast.FloorDiv, ast.Mod),
        ):
            if right == 0:
                raise ZeroDivisionError("Division by zero")

        # ----------------------------------------------------
        # CALCULATE
        # ----------------------------------------------------

        try:
            result = operator_function(left, right)

        except ZeroDivisionError:
            raise

        except OverflowError as exc:
            raise ValueError("Result is too large") from exc

        except Exception as exc:
            raise ValueError("Invalid calculation") from exc

        return _check_value(result)

    raise ValueError("Unsupported expression")


# ============================================================
# EXPRESSION PARSING
# ============================================================

def _validate_expression_text(expression: str) -> str:
    """
    Expression text basic validation।
    """

    if not isinstance(expression, str):
        raise ValueError("Expression must be text")

    expression = expression.strip()

    if not expression:
        raise ValueError("Expression is empty")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long")

    # Only mathematical characters.
    if not all(
        char.isdigit()
        or char in "+-*/%(). "
        for char in expression
    ):
        raise ValueError("Invalid characters in expression")

    if not any(char.isdigit() for char in expression):
        raise ValueError("No number found")

    return expression


def _parse_expression(expression: str) -> ast.Expression:
    """
    Expression → AST।
    """

    try:
        tree = ast.parse(expression, mode="eval")

    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(
            "Invalid mathematical expression"
        ) from exc

    if not isinstance(tree, ast.Expression):
        raise ValueError("Invalid expression")

    _count_nodes(tree)

    return tree


# ============================================================
# RESULT FORMATTING
# ============================================================

def _format_result(value: Number) -> Number:
    """
    5.0 → 5
    5.25 → 5.25

    Fractional result কখনো silently round করা হবে না।
    """

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Result is not finite")

        # Exact integer-valued float।
        if value.is_integer():
            return int(value)

    return value


# ============================================================
# PUBLIC CALCULATOR
# ============================================================

def calculate(expression: str) -> Number:
    """
    Safe mathematical calculator।

    Examples:
        calculate("2 + 3")
        calculate("(10 + 5) * 2")
        calculate("10 / 4")

    Raises:
        ValueError
    """

    expression = _validate_expression_text(expression)

    tree = _parse_expression(expression)

    try:
        result = _evaluate(tree.body)

    except ZeroDivisionError as exc:
        # Public API-তে traceback না দেখিয়ে
        # একটি সাধারণ ValueError দেওয়া হবে।
        raise ValueError("Cannot divide by zero") from None

    return _format_result(result)


# ============================================================
# PUBLIC VALIDATION HELPER
# ============================================================

def is_math_expression(expression: str) -> bool:
    """
    Expression valid mathematical expression কি না।
    """

    try:
        calculate(expression)
        return True

    except Exception:
        return False


__all__ = [
    "calculate",
    "is_math_expression",
]