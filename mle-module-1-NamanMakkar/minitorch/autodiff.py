from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Dict

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    x1 : Any = [i for i in vals]
    x1[arg] += epsilon
    x2 : Any = [i for i in vals]
    x2[arg] -= epsilon

    return (f(*x1) - f(*x2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    marked : Dict[int, Variable] = dict()

    def visit(variable: Variable) -> None:
        if (variable.unique_id in marked.keys() or variable.is_constant()) :
            return

        if not variable.is_leaf():
            for parent in variable.parents:
                visit(parent)

        marked[variable.unique_id] = variable

    visit(variable)
    visited_anti_top : Any = list(marked.values())
    visited_top : Iterable[Variable] = visited_anti_top[::-1]

    return visited_top


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    topologically_sorted_variables : Iterable[Variable] = topological_sort(variable)
    variables_with_derivs : Dict[int, Any] = {variable.unique_id: deriv}

    for var in topologically_sorted_variables:
        if var.is_leaf():
            var.accumulate_derivative(variables_with_derivs[var.unique_id])
        else:
            derivs_back = var.chain_rule(variables_with_derivs[var.unique_id])
            for scalar, der in derivs_back:
                if (scalar.unique_id in variables_with_derivs.keys()):
                    variables_with_derivs[scalar.unique_id] += der
                else:
                    variables_with_derivs[scalar.unique_id] = der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
