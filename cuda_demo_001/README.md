# cuda_demo_001

## 1 实现在GPU上进行加法运算

构建一个.cuh文件进行cuda_add函数的声明，在.cu文件中进行函数的定义。最后在.cpp文件中进行调用。
## 2 实现CUDA初始化过程

首先获取设备的cuda设备个数，如果没有支持cuda的设备，那么返回1。1是device0，是一个仿真设备。不支持cuda1.0以上的版本。因此可以使用prop.major进行判断，如果遍历完所有的设备依然没有大于cuda1.0的，那么就是没有支持的实际设备。使用cudaGetDevicePropertoes可以获取cuda的版本、设备的名称、内存的大小、最大的线程数、执行单元的频率等。如果cuda版本是7.5的, prop.major = 7, prop.minor = 5。

## 3 实现CUDA一个简单的核函数