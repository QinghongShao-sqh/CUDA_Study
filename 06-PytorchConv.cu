import time

import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


width = 1000;
height = 1000;
#img =torch.ones([width,height])
img =torch.randn([width,height])
img = img.to(device)

kernel = torch.tensor([[-1.0, 0.0, 1.0],
                       [-1.0, 0.0, 1.0],
                       [-1.0, 0.0, 1.0]])


#input = torch.reshape(input, (1, 1, 5, 5))
img = torch.reshape(img, (1, 1, width, height))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
kernel = kernel.to(device)

output = F.conv2d(img, kernel, stride=1).to(device)
# torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
# 返回的是 s ， 乘1000 为ms
start = time.perf_counter()

# output = F.conv2d(img, kernel, stride=1).to(device)
output = F.conv2d(img, kernel, stride=1).to(device)

end = time.perf_counter()

print("startime:",start)
print("endtime:",end)
print("total:",end-start)
print("output:size===>",output.shape)
print("output tensor:",output)

