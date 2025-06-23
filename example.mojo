from math import e, log, floor

def exp(x:Float64) -> Float64:
  return e ** x

def sum(values:List[Float64]) -> Float64:
  total:Float64 = 0.0
  for value in values:
    total += value
  return total

struct Tensor(Writable, Stringable, Movable, Copyable):
  var data: Float64
  var _prevs: List[Tensor]
  var grad: Float64
  var _backward: Tuple[Optional[Tensor], String]

  def __init__(out self, data:Float64, _prevs:List[Tensor]=[]):
    self.data = data
    self._prevs = _prevs
    self.grad = 0.0
    self._backward = (None, "__init__")

  def __add__(self, other:Tensor)-> Tensor:
    out = Tensor(self.data + other.data, _prevs=[self, other])
    out._backward = (out, "__add__")
    return out

  def __mul__(self, other:Tensor) -> Tensor:
    out = Tensor(self.data * other.data, _prevs=[self, other])
    out._backward = (out, "__mul__")
    return out

  def __pow__(self, other:Tensor)-> Tensor:
    out =  Tensor(self.data ** other.data, _prevs=[self, other])
    out._backward = (out, "__pow__")
    return out

  def __sub__(self, other:Tensor) -> Tensor:
    return self + (-other)

  def __neg__(self) -> Tensor:
    return self * Tensor(-1)

  def __truediv__(self, other:Tensor)-> Tensor:
    return self * (other ** Tensor(-1))

  def __floordiv__(self, other:Tensor)-> Tensor:
    out = self * (other ** Tensor(-1))
    out.data = floor(out.data)
    return out

  def __mod__(self, other:Tensor)-> Tensor:
    out = Tensor(self.data % other.data, _prevs=[self, other])
    out._backward = (out, "__mod__")
    return out

  def __ge__(self, other:Tensor)->Tensor:
    out = Tensor(1 if self.data >= other.data else 0, _prevs=[self, other])
    out._backward = (out, "__ge__")
    return out

  def __gt__(self, other:Tensor)->Tensor:
    out = Tensor(1 if self.data > other.data else 0, _prevs=[self, other])
    out._backward = (out, "__gt__")
    return out

  def __le__(self, other:Tensor)->Tensor:
    out = Tensor(1 if self.data <= other.data else 0, _prevs=[self, other])
    out._backward = (out, "__le__")
    return out

  def __lt__(self, other:Tensor)->Tensor:
    out = Tensor(1 if self.data < other.data else 0, _prevs=[self, other])
    out._backward = (out, "__lt__")
    return out

  def __eq__(self, other:Tensor) -> Tensor:
    out = Tensor(1 if self.data == other.data else 0, _prevs=[self, other])
    return out

  def __is__(self, other:Tensor) -> Bool:
    return self.data == other.data

  def _add_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for addition")
    _self, other = _out._prevs[0] , _out._prevs[1]
    _self.grad += _out.grad
    other.grad += _out.grad

  def _mul_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for multiplication")
    _self, other = _out._prevs[0] , _out._prevs[1]
    _self.grad += _out.grad * other.data
    other.grad += _out.grad * _self.data

  def _pow_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for power operation")
    _self, other = _out._prevs[0] , _out._prevs[1]
    _self.grad += _out.grad * other.data * (self.data ** (other.data - 1))
    other.grad += _out.grad * (self.data ** other.data) * log(self.data)
    return

  def _mod_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for modulo operation")
    _self, other = _out._prevs[0] , _out._prevs[1]
    if other.data != 0:
      _self.grad += _out.grad
      other.grad += _out.grad * -(_self.data // other.data + 1)

  def _ge_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for greater than or equal operation")
    _self, other = _out._prevs[0] , _out._prevs[1]
    if _self.data >= other.data:
      _self.grad += _out.grad
    else:
      other.grad += _out.grad

  def _gt_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for greater than operation")
    _self, other = _out._prevs[0] , _out._prevs[1]
    if _self.data > other.data:
      _self.grad += _out.grad
    else:
      other.grad += _out.grad

  def _le_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for less than or equal operation")
    _self, other = _out._prevs[0] , _out._prevs[1]
    if _self.data <= other.data:
      _self.grad += _out.grad
    else:
      other.grad += _out.grad

  def _lt_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for less than operation")
    _self, other = _out._prevs[0] , _out._prevs[1]
    if _self.data < other.data:
      _self.grad += _out.grad
    else:
      other.grad += _out.grad

  def _eq_back(self, _out: Tensor):
    debug_assert(len(_out._prevs) == 2, "Expected two predecessors for equality operation")
    _self, other = _out._prevs[0] , _out._prevs[1]
    if _self.data == other.data:
      _self.grad += _out.grad
    else:
      other.grad += _out.grad

  def set_grad(mut self, grad:Float64):
      self.grad = grad

  fn write_to[W: Writer](self, mut writer: W):
      writer.write(
          "Tensor(", self.data, ", prevs_len=", len(self._prevs), ")"
      )

  fn __str__(self) -> String:
      return String(self)

fn tensor_preorder_traversal(
    node: Tensor, mut visited: List[Tensor]) -> List[Tensor]:

    for i in range(len(visited)):
        if visited[i].data == node.data:
            return []
    visited.append(node)

    var result:List[Tensor] = List[Tensor]()
    result.append(node)

    for prev in node._prevs:
        sub = tensor_preorder_traversal(prev, visited)
        result.extend(sub)

    return result

def relu(x:Tensor) -> Tensor:
  out = Tensor(max(0, x.data), _prevs=[x])
  out._backward = (out, "relu")
  return out

def relu_back(_out: Tensor):
    _self = _out._prevs[0]
    if _self.data > 0:
        _self.grad += _out.grad
    else:
        _self.grad += 0

def select_op_back(_out:Tensor, op:String):
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

def backpropagate(transversed:List[Tensor]):
    # Set initial gradient to 1 for the final output
    if len(transversed) > 0:
      var first = transversed[0]
      first.set_grad(1.0)
    
    # Backpropagate through the graph
    for node in transversed:
        if len(node._prevs) > 0:
          if node._backward[0] is not None:
            select_op_back(node._backward[0].value(), node._backward[1])

def main():
  tensor1 = Tensor(3)
  tensor2 = Tensor(6)
  tensor_sum = tensor1 + tensor2
  print("The sum of tensor1 and tensor2 is:", tensor_sum)

  tensor_diff = tensor1 - tensor_sum
  print("The difference of tensor1 and tensor_sum is:", tensor_diff)

  tensor_neg = -tensor_diff
  print("The negation of tensor_diff is:", tensor_neg)

  final_tensor = tensor_neg * Tensor(2) + Tensor(5)
  print("The final tensor is:", final_tensor)
  
  var tensor_list:List[Tensor] = List[Tensor]()
  result = tensor_preorder_traversal(tensor_sum, tensor_list)
  print("Traversing the tensor graph length ", len(result), ":")

  for i in range(len(result)):
    print(i, result[i])

  # Backpropagate through the graph
  backpropagate(result)
  
  print("\nGradients after backpropagation:")
  for i in range(len(result)):
    print("Gradient of node", i, ":", result[i].grad)
    
