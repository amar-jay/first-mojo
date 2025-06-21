from math import e
def exp(x:Float64) -> Float64:
  return e ** x
struct Tensor(Stringable, Movable, Copyable, Movable):
  var data: Float64
  var _prevs: List[Tensor]
  var grad: Float64 = 0.0
  var _backward: () -> Void

  def __init__(out self, data:Float64, _prevs:List[Tensor]=[]):
    self.data = data
    self._prevs = _prevs
    self.grad = 0.0

  def __add__(self, other:Tensor)-> Tensor:
    out = Tensor(self.data + other.data, _prevs=[self, other])
    def _backward():
      self.grad += out.grad
      outher.grad += out.grad
    out._backward = _backward
    return out

  def __mul__(self, other:Tensor) -> Tensor:
    out = Tensor(self.data * other.data, _prevs=[self, other])
    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data
    out._backward = _backward
    return out

  def __pow__(self, other:Tensor)-> Tensor:
    out =  Tensor(self.data ** other.data, _prevs=[self, other])
    def _backward():
      self.grad += out.grad * other.data * (self.data ** (other.data - 1))
      other.grad += out.grad * (self.data ** other.data) * log(self.data)
    out._backward = _backward^
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
    def _backward():
      self.grad += out.grad
      other.grad += out.grad * (self.data // other.data)
    out._backward = _backward^
    return out

  def __ge__(self, other:Tensor)->Tensor:
    out = Tensor(1 if self.data >= other.data else 0, _prevs=[self, other])
    def _backward():
      if self.data >= other.data:
        self.grad += out.grad
      else:
        other.grad += out.grad

    out._backward = _backward^
    return out

  def __gt__(self, other:Tensor)->Tensor:
    out = Tensor(1 if self.data > other.data else 0, _prevs=[self, other])
    def _backward():
      if self.data > other.data:
        self.grad += out.grad
      else:
        other.grad += out.grad
    out._backward = _backward^
    return out

  def __le__(self, other:Tensor)->Tensor:
    out = Tensor(1 if self.data <= other.data else 0, _prevs=[self, other])
    def _backward():
      if self.data <= other.data:
        self.grad += out.grad
      else:
        other.grad += out.grad
    out._backward = _backward^
    return out

  def __lt__(self, other:Tensor)->Tensor:
    out = Tensor(1 if self.data < other.data else 0, _prevs=[self, other])
    def _backward():
      if self.data < other.data:
        self.grad += out.grad
      else:
        other.grad += out.grad
    out._backward = _backward^
    return out

  def __eq__(self, other:Tensor) -> Tensor:
    out = Tensor(1 if self.data == other.data else 0, _prevs=[self, other])
    def _backward():
      if self.data == other.data:
        self.grad += out.grad
      else:
        other.grad += out.grad
    out._backward = _backward^
    return out

  def __is__(self, other:Tensor) -> Bool:
    return self.data == other.data

#  def __getitem__(self, key:Int):
#    return self.data[key]

  fn __str__(self) -> String:
      return String("Tensor(" , self.data, ", prevs_len=", len(self._prevs), ")")

fn tensor_pre_order_traversal(
    node: Tensor,mut visited: List[Tensor]) -> List[Tensor]:

    for i in range(len(visited)):
        if visited[i].data == node.data:
            return []
    visited.append(node)

    var result:List[Tensor] = List[Tensor]()
    result.append(node)

    for prev in node._prevs:
        sub = tensor_pre_order_traversal(prev, visited)
        result.extend(sub)

    return result

def relu(x:Tensor) -> Tensor:
  out = Tensor(max(0, x.data), _prevs=[x])
  def _backward():
    x.grad += out.grad if x.data > 0 else 0
  out._backward = _backward
  return out

def sigmoid(x:Tensor) -> Tensor:
  out = Tensor(1 / (1 + exp(-x.data)), _prevs=[x])
  def _backward():
    x.grad += out.grad * out.data * (1 - out.data)
  out._backward = _backward
  return out

def tanh(x:Tensor) -> Tensor:
  out = Tensor((exp(x.data) - exp(-x.data)) / (exp(x.data) + exp(-x.data)), _prevs=[x])
  def _backward():
    x.grad += out.grad * (1 - out.data ** 2)
  out._backward = _backward
  return out

def softmax(x:List[Tensor]) -> List[Tensor]:
  exp_values = [exp(t.data) for t in x]
  sum_exp = 0
  for ev in exp_values:
    sum_exp += ev
  out = [Tensor(ev / sum_exp, _prevs=[t]) for ev, t in zip(exp_values, x)]
  def _backward():
    for i in range(len(out)):
      out[i].grad += out[i].data * (1 - out[i].data) * sum([t.grad for t in out])
  for t in out:
    t._backward = _backward
  return out

def cross_entropy_loss(predictions:List[Tensor], targets:List[Tensor]) -> Tensor:
  loss = -sum([t.data * log(p.data) for p, t in zip(predictions, targets)])
  out = Tensor(loss, _prevs=predictions + targets)
  def _backward():
    for i in range(len(predictions)):
      predictions[i].grad += out.grad * (predictions[i].data - targets[i].data)
  out._backward = _backward
  return out
fn backpropagate(transversed:List[Tensor]) -> List[Tensor]:
    var gradients:List[Tensor] = List[Tensor]()
    for node in transversed:
        if len(node._prevs) == 0:
            continue
        for prev in node._prevs:
            prev.grad = 0.0  # Reset gradients for each node
        for prev in node._prevs:
          prev._backward()
        for prev in node._prevs:
            gradients.append(prev.grad)
    return gradients

def main():
  tensor1 = Tensor(3)
  tensor2 = Tensor(6)
  tensor_sum = tensor1 + tensor2
  print("The sum of tensor1 and tensor2 is:", tensor_sum.__str__())

  tensor_diff = tensor1 - tensor_sum
  print("The difference of tensor1 and tensor2 is:", tensor_diff.__str__())

  tensor_neg = -tensor_diff
  print("The negation of tensor1 is:", tensor_neg.__str__())

  final_tensor = tensor_neg * Tensor(2) + Tensor(5)
  print("The final tensor is:", final_tensor.__str__())
  var tensor_list:List[Tensor] = List[Tensor]()
  result = tensor_pre_order_traversal(final_tensor, tensor_list)
  print("Traversing the tensor graph length ", len(result), ":")

  for i in range(len(result)):
    print(i, result[i].__str__())

  # backpropagate(result)
  gradients = backpropagate(result)
  for i in range(len(gradients)):
    print("Gradient of node", i, ":", gradients[i])

