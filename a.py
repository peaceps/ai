import torch
tensor1 = torch.tensor([5.5, 3])
tensor2 = tensor1.new_ones(5, 3, dtype=torch.double)
tensor3 = torch.randn_like(tensor2, dtype=torch.float)
tensor4 = torch.rand(5, 3)
print('tensor3 + tensor4= ', tensor3 + tensor4)
print('tensor3 + tensor4= ', torch.add(tensor3, tensor4))

result = torch.empty(5, 3)
torch.add(tensor3, tensor4, out=result)
print('add result= ', result)

tensor3.add_(tensor4)
print('tensor3= ', tensor3)
tensor3.t_()