
def conv2d(size_in,kernel_size,stride=1,padding=0):
    out = int(((size_in+2*padding-(kernel_size-1)-1)/stride)+1)
    print(f"output size conv2d : {out}")
    return out

def maxpool2d(size_in,kernel_size,stride=1,padding=0):
    out = int(((size_in+2*padding-(kernel_size-1)-1)/stride)+1)
    print(f"output size maxpool2d : {out}")
    return out

input_size = 80

s = conv2d(input_size,11,4,2)
s = maxpool2d(s,3,2)
s = conv2d(s,5,1,2)
s = maxpool2d(s,3,2)
s = conv2d(s,3,1,1)
s = conv2d(s,3,1,1)
s = conv2d(s,3,2,2)
s = maxpool2d(s,3,2)



