"""
This code is an implementation of autodiff engine similar to karpathy/tinygrad with some differences to fit the Mojo syntax and features.
It was first written in Mojo and then translated to Python by ChatGPT for testing purposes since I'm having memory issues running Mojo code.
This is a temporary measure until Mojo's memory issues are resolved.
"""

import math
from typing import List, Optional, Tuple, final

class Tensor:
    def __init__(self, data: float, _prevs: List["Tensor"] = []):
        self.data: float = data
        self._prevs: List["Tensor"] = _prevs
        self.grad: float = 0.0
        self._backward: Tuple[Optional["Tensor"], str] = (None, "__init__")

    def __add__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data + other.data, [self, other])
        out._backward = (out, "__add__")
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data * other.data, [self, other])
        out._backward = (out, "__mul__")
        return out

    def __pow__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data ** other.data, [self, other])
        out._backward = (out, "__pow__")
        return out

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self + (-other)

    def __neg__(self) -> "Tensor":
        return self * Tensor(-1)

    def __truediv__(self, other: "Tensor") -> "Tensor":
        return self * (other ** Tensor(-1))

    def __floordiv__(self, other: "Tensor") -> "Tensor":
        out = self * (other ** Tensor(-1))
        out.data = math.floor(out.data)
        return out

    def __mod__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data % other.data, [self, other])
        out._backward = (out, "__mod__")
        return out

    def __ge__(self, other: "Tensor") -> "Tensor":
        out = Tensor(1.0 if self.data >= other.data else 0.0, [self, other])
        out._backward = (out, "__ge__")
        return out

    def __gt__(self, other: "Tensor") -> "Tensor":
        out = Tensor(1.0 if self.data > other.data else 0.0, [self, other])
        out._backward = (out, "__gt__")
        return out

    def __le__(self, other: "Tensor") -> "Tensor":
        out = Tensor(1.0 if self.data <= other.data else 0.0, [self, other])
        out._backward = (out, "__le__")
        return out

    def __lt__(self, other: "Tensor") -> "Tensor":
        out = Tensor(1.0 if self.data < other.data else 0.0, [self, other])
        out._backward = (out, "__lt__")
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return False
        return self.data == other.data

    def __str__(self) -> str:
        return f"Tensor({self.data}, prevs_len={len(self._prevs)})"

    def set_grad(self, grad: float):
        self.grad = grad

    def _add_back(self, _out: "Tensor"):
        a, b = _out._prevs
        a.grad += _out.grad
        b.grad += _out.grad

    def _mul_back(self, _out: "Tensor"):
        a, b = _out._prevs
        a.grad += _out.grad * b.data
        b.grad += _out.grad * a.data

    def _pow_back(self, _out: "Tensor"):
        base, exp_tensor = _out._prevs
        base.grad += _out.grad * exp_tensor.data * (base.data ** (exp_tensor.data - 1))
        exp_tensor.grad += _out.grad * (base.data ** exp_tensor.data) * math.log(base.data)

    def _mod_back(self, _out: "Tensor"):
        a, b = _out._prevs
        if b.data != 0:
            a.grad += _out.grad
            b.grad += _out.grad * -(a.data // b.data + 1)

    def _ge_back(self, _out: "Tensor"):
        a, b = _out._prevs
        if a.data >= b.data:
            a.grad += _out.grad
        else:
            b.grad += _out.grad

    def _gt_back(self, _out: "Tensor"):
        a, b = _out._prevs
        if a.data > b.data:
            a.grad += _out.grad
        else:
            b.grad += _out.grad

    def _le_back(self, _out: "Tensor"):
        a, b = _out._prevs
        if a.data <= b.data:
            a.grad += _out.grad
        else:
            b.grad += _out.grad

    def _lt_back(self, _out: "Tensor"):
        a, b = _out._prevs
        if a.data < b.data:
            a.grad += _out.grad
        else:
            b.grad += _out.grad

    def _eq_back(self, _out: "Tensor"):
        a, b = _out._prevs
        if a.data == b.data:
            a.grad += _out.grad
        else:
            b.grad += _out.grad


def tensor_preorder_traversal(node: Tensor, visited: List[Tensor]) -> List[Tensor]:
    if any(n.data == node.data for n in visited):
        return []

    visited.append(node)
    result = [node]
    for prev in node._prevs:
        result.extend(tensor_preorder_traversal(prev, visited))
    return result

def relu(x: Tensor) -> Tensor:
    out = Tensor(max(0, x.data), [x])
    out._backward = (out, "relu")
    return out

def relu_back(_out: Tensor):
    x = _out._prevs[0]
    x.grad += _out.grad if x.data > 0 else 0

def select_op_back(_out: Tensor, op: str):
    if op == "__add__":
        _out._add_back(_out)
    elif op == "__mul__":
        _out._mul_back(_out)
    elif op == "__pow__":
        _out._pow_back(_out)
    elif op == "__mod__":
        _out._mod_back(_out)
    elif op == "__ge__":
        _out._ge_back(_out)
    elif op == "__gt__":
        _out._gt_back(_out)
    elif op == "__le__":
        _out._le_back(_out)
    elif op == "__lt__":
        _out._lt_back(_out)
    elif op == "__eq__":
        _out._eq_back(_out)
    elif op == "relu":
        relu_back(_out)

def backpropagate(transversed: List[Tensor]):
    if transversed:
        transversed[0].set_grad(1.0)

    for node in transversed:
        if node._prevs and node._backward[0] is not None:
            select_op_back(node._backward[0], node._backward[1])

def main():
    tensor1 = Tensor(3.1)
    tensor2 = Tensor(6.2)
    tensor_sum = tensor1 + tensor2
    print("The sum of tensor1 and tensor2 is:", tensor_sum)

    tensor_diff = tensor1 - tensor_sum
    print("The difference of tensor1 and tensor_sum is:", tensor_diff)

    tensor_neg = -tensor_diff
    print("The negation of tensor_diff is:", tensor_neg)

    final_tensor = tensor_neg * Tensor(2.3) + Tensor(5.4) / Tensor(2.5)
    print("The final tensor is:", final_tensor)

    tensor_list: List[Tensor] = []
    result = tensor_preorder_traversal(final_tensor, tensor_list)
    print("Traversing the tensor graph length", len(result), ":")

    for i, tensor in enumerate(result):
        print(i, tensor)

    backpropagate(result)

    print("\nGradients after backpropagation:")
    for i, tensor in enumerate(result):
        print(f"Gradient of node {i}:", tensor.grad)

if __name__ == "__main__":
    #TODO: compare it with PyTorch and see if it matches and at what precision
    main()
