# 时间测试

使用cuda自带的事件机制进行时间测试。大致过程如下：

    # 创建事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    # 记录时间
    cudaEventRecord(start, 0);
    ... // 发生的事件
    cudaEventRecord(stop, 0);

    # 同步事件（否则时间差会出现异常）
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    # 计算时间（不能使用double，只能使用float类型）
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    # 销毁事件对象
    cudaEventDestroy(start);
    cudaEventDestroy(stop);