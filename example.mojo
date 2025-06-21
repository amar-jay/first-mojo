struct Tensor(Stringable, Movable, Copyable, Movable):
  var data: Float64
  var _prevs: List[Tensor]

  def __init__(out self, data:Float64, _prevs:List[Tensor]=[]):
    self.data = data
    self._prevs = _prevs

  def __add__(self, other:Tensor)-> Tensor:
    return Tensor(self.data + other.data, _prevs=[self, other])

  def __sub__(self, other:Tensor) -> Tensor:
    return Tensor(self.data + other.data, _prevs=[self, other])

  def __neg__(self) -> Tensor:
    return Tensor(-self.data, _prevs=[self])

  def __mul__(self, other:Tensor) -> Tensor:
    return Tensor(self.data * other.data, _prevs=[self, other])

  def __truediv__(self, other:Tensor)-> Tensor:
    return Tensor(self.data / other.data, _prevs=[self, other])

  def __floordiv__(self, other:Tensor)-> Tensor:
    return Tensor(self.data // other.data, _prevs=[self, other])

  def __mod__(self, other:Tensor)-> Tensor:
    return Tensor(self.data % other.data, _prevs=[self, other])

  def __pow__(self, other:Tensor)-> Tensor:
    return Tensor(self.data ** other.data, _prevs=[self, other])

  def __ge__(self, other:Tensor)->Tensor:
    return Tensor(1 if self.data >= other.data else 0, _prevs=[self, other])

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

fn backpropagation(transversed:List[Tensor], grad:Float64) -> List[Tensor]:
    var gradients:List[Tensor] = List[Tensor]()
    for node in transversed:
        if len(node._prevs) == 0:
            continue
        for prev in node._prevs:
            gradients.append(prev * grad)
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

