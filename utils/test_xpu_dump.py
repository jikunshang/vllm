from tensor_dump import TensorDumper

import torch
import intel_extension_for_pytorch as ipex

def test():
    dumper = TensorDumper("abcde",3)
    t1 = torch.tensor([1,2,3]).to('xpu')
    dumper.append(t1)
    t2 = torch.tensor([3,2,1]).to('xpu')
    dumper.append(t2)
    

def read():
    path = "/root/data/xpu_abcde_2"
    xput_t = torch.load(path)
    print(xput_t)

if __name__ == "__main__":
    test()
    read()