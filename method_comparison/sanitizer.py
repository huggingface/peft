import ast

import pandas as pd


def _evaluate_node(df, node):
    """
    Recursively evaluates an AST node to generate a pandas boolean mask.
    """
    # Base Case: A simple comparison like 'price > 100'
    if isinstance(node, ast.Compare):
        if not isinstance(node.left, ast.Name):
            raise ValueError("Left side of comparison must be a column name.")
        col = node.left.id
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        if len(node.ops) > 1:
            raise ValueError("Chained comparisons like '10 < price < 100' are not supported.")

        op_node = node.ops[0]
        val_node = node.comparators[0]
        try:
            value = ast.literal_eval(val_node)
        except ValueError:
            raise ValueError("Right side of comparison must be a literal (number, string, list).")

        operator_map = {
            ast.Gt:    lambda c, v: df[c] > v,
            ast.GtE:   lambda c, v: df[c] >= v,
            ast.Lt:    lambda c, v: df[c] < v,
            ast.LtE:   lambda c, v: df[c] <= v,
            ast.Eq:    lambda c, v: df[c] == v,
            ast.NotEq: lambda c, v: df[c] != v,
            ast.In:    lambda c, v: df[c].isin(v),
            ast.NotIn: lambda c, v: ~df[c].isin(v)
        }
        op_type = type(op_node)
        if op_type not in operator_map:
            raise ValueError(f"Unsupported operator '{op_type.__name__}'.")
        return operator_map[op_type](col, value)

    # Recursive Step: "Bitwise" operation & and | (the same as boolean operations)
    elif isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitOr):
            return _evaluate_node(df, node.left) | _evaluate_node(df, node.right)
        elif isinstance(node.op, ast.BitAnd):
            return _evaluate_node(df, node.left) & _evaluate_node(df, node.right)

    # Recursive Step: A boolean operation like '... and ...' or '... or ...'
    elif isinstance(node, ast.BoolOp):
        op_type = type(node.op)
        # Evaluate the first value in the boolean expression
        result = _evaluate_node(df, node.values[0])
        # Combine it with the rest of the values based on the operator
        for i in range(1, len(node.values)):
            if op_type is ast.And or op_type is ast.BitAnd:
                result &= _evaluate_node(df, node.values[i])
            elif op_type is ast.Or or op_type is ast.BitOr:
                result |= _evaluate_node(df, node.values[i])
        return result

    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.Not):
            raise ValueError("Only supported unary op is negation.")
        return ~_evaluate_node(df, node.operand)

    # If the node is not a comparison or boolean op, it's an unsupported expression type
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def parse_and_filter(df, filter_str):
    """
    Filters a pandas DataFrame using a string expression parsed by AST.
    This is done to avoid the security vulnerables that `DataFrame.query`
    brings (arbitrary code execution).

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        filter_str (str): A string representing a filter expression.
                          e.g., "price > 100 and stock < 50"
                          Supported operators: >, >=, <, <=, ==, !=, in, not in, and, or.

    Returns:
        pd.Series: A boolean Series representing the filter mask.
    """
    if not filter_str:
        return pd.Series([True] * len(df), index=df.index)

    try:
        # 'eval' mode ensures the source is a single expression.
        tree = ast.parse(filter_str, mode='eval')
        expression_node = tree.body
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid filter syntax: {e}")

    # The recursive evaluation starts here
    mask = _evaluate_node(df, expression_node)
    return mask
