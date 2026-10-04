"""Safe arithmetic service for Hello AI."""

import ast
import math
import operator
import re


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_EXPRESSION_LENGTH = 200
MAX_ABS_VALUE = 10**100
MAX_EXPONENT = 100
MAX_AST_NODES = 100


def _check_number(value):
    """Reject non-finite or excessively large numeric values."""

    if isinstance(value, bool):
        raise ValueError("Boolean values are not allowed")

    if not isinstance(value, (int, float)):
        raise ValueError("Only numeric results are allowed")

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result must be finite")

    if abs(value) > MAX_ABS_VALUE:
        raise ValueError("Number is too large")

    return value


def _evaluate(node):
    """Evaluate only explicitly permitted arithmetic syntax."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("Boolean values are not allowed")

        if isinstance(node.value, (int, float)):
            return _check_number(node.value)

        raise ValueError("Only numbers are allowed")

    if isinstance(node, ast.BinOp):
        operation = BINARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Operator is not allowed")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError("Exponent is too large")

            if isinstance(left, int) and isinstance(right, int):
                if right > 0 and abs(left) > 1:
                    # Check the approximate result size before
                    # calculating a potentially enormous power.
                    if left.bit_length() * right > 400:
                        raise ValueError("Result is too large")

        if isinstance(node.op, ast.Mult):
            if left != 0 and right != 0:
                if abs(left) > MAX_ABS_VALUE / abs(right):
                    raise ValueError("Result is too large")

        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if right == 0:
                raise ValueError("Division by zero")

        try:
            result = operation(left, right)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("Invalid arithmetic operation") from exc

        if isinstance(result, complex):
            raise ValueError("Complex results are not supported")

        return _check_number(result)

    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unary operator is not allowed")

        return _check_number(operation(_evaluate(node.operand)))

    raise ValueError("Expression contains unsupported syntax")


def calculate(expression: str):
    """Calculate a basic arithmetic expression safely."""

    if not isinstance(expression, str):
        raise ValueError("Expression must be text")

    expression = expression.strip()

    if not expression:
        raise ValueError("Expression is empty")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long")

    if not re.fullmatch(r"[0-9+\-*/%().\s]+", expression):
        raise ValueError("Expression contains unsupported characters")

    try:
        tree = ast.parse(expression, mode="eval")

        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise ValueError("Expression is too complex")

        result = _evaluate(tree.body)

    except (SyntaxError, OverflowError, RecursionError) as exc:
        raise ValueError("Invalid arithmetic expression") from exc

    if isinstance(result, float) and result.is_integer():
        return int(result)

    return result