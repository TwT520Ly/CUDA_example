# cuda_demo_001

## 1 实现在GPU上进行加法运算

构建一个.cuh文件进行cuda_add函数的声明，在.cu文件中进行函数的定义。最后在.cpp文件中进行调用。

## 2 实现CUDA初始化过程

首先获取设备的cuda设备个数，如果没有支持cuda的设备，那么返回1。1是device0，是一个仿真设备。不支持cuda1.0以上的版本。因此可以使用prop.major进行判断，如果遍历完所有的设备依然没有大于cuda1.0的，那么就是没有支持的实际设备。使用cudaGetDevicePropertoes可以获取cuda的版本、设备的名称、内存的大小、最大的线程数、执行单元的频率等。如果cuda版本是7.5的, prop.major = 7, prop.minor = 5。

## 3 实现CUDA一个简单的核函数

实现一堆随机数的立方和相加的过程，需要首先在CPU上进行数据的初始化，然后在GPU（device）上创建内存区域（RAM），将数据拷贝到显存中。使用CUDA程序进行计算，然后将结果再传递回host端。使用cudaMalloc进行内存分配，然后使用cudaMemcpy进行内存拷贝，不过在进行拷贝的过程中，需要指定拷贝的方向。
对于在显卡执行的函数，不能有返回值，因此要通过指针的方式进行传递。

## 4 测试时间
CLOCKS_PER_SEC是系统一秒钟的时钟频率，clock_t是长整型类型。使用clock函数进行时间测试，但是需要除以CPU与GPU的时钟频率才可以得到大致的时间。在进行时间测试的时候，CPU使用clock测试是没有问题的，但是GPU使用clock出现了负值的情况。在kernel占用时间较小的时候，不会出现该问题，但是只要占用时间长一点，就会出错。将clock_t类型换成unsigned long long类型就没有问题了。怀疑是溢出的问题，在Linux系统下面出现了溢出时间数据范围的bug。

## 5 设备属性
使用cudaGetDeviceProperties可以获取到设备的一个属性集合。返回值cudaDeviceProp是一个结构体类型。属性clockRate单位是千兆赫兹。
