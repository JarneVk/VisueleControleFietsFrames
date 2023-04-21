
def conv2d(size_in,kernel_size,stride=1,padding=0):
    out = int(((size_in+2*padding-(kernel_size-1)-1)/stride)+1)
    print(f"output size conv2d : {out}")
    return out

def maxpool2d(size_in,kernel_size,stride=1,padding=0):
    out = int(((size_in+2*padding-(kernel_size-1)-1)/stride)+1)
    print(f"output size maxpool2d : {out}")
    return out

def convT2d(size_in,kernel_size,stride=1,padding=0,output_padding=0):
    out = (size_in-1) * stride - 2 * padding * (kernel_size-1) + output_padding + 1
    print(f"output size convT2d : {out}")
    return out

input_size = 80

# s = conv2d(input_size,5,4,2)
# s = conv2d(s,5,1)
# s = conv2d(s,3,2,2)
# s = conv2d(s,3,1,1)
# s = conv2d(s,3,2,2)
# #decoder
# s = convT2d(s,3,2,2,1)
# s = conv2d(s,3,padding=1)
# s = convT2d(s,3,2,2,1)
# s = conv2d(s,5,padding=1)
# s = convT2d(s,5,4,4,1)

s = conv2d(input_size,11,4,2)
s = conv2d(s,3,1)
s = conv2d(s,5,2,2)
s = conv2d(s,3,1,1)
s = conv2d(s,3,2,2)

s = convT2d(s,3,2,2,1)
s = conv2d(s,3,padding=1)
s = convT2d(s,5,2,2,1)
s = conv2d(s,3,padding=1)
s = convT2d(s,11,4,4,1)




