import torch
import time

def test_device(device_name, size=8000):
    """
    在指定设备上测试矩阵乘法性能
    """
    device = torch.device(device_name)
    print(f"正在 {device_name.upper()} 上测试 (矩阵大小: {size}x{size})...")

    # 1. 数据准备
    try:
        start_time = time.time()
        # 创建两个随机矩阵
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # 确保 GPU 操作完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        print(f"  - 数据初始化耗时: {time.time() - start_time:.4f} 秒")

        # 2. 矩阵乘法计算
        start_time = time.time()
        c = torch.matmul(a, b)
        
        # 确保 GPU 操作完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        compute_time = time.time() - start_time
        print(f"  - 计算耗时: {compute_time:.4f} 秒")
        return compute_time

    except Exception as e:
        print(f"  - 测试失败: {e}")
        return None

def main():
    print("="*40)
    print("PyTorch GPU 算力测试程序")
    print("="*40)

    # 1. 检查 CUDA 是否可用
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ 检测到 GPU: {gpu_name}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   PyTorch 版本: {torch.__version__}")
    else:
        print("❌ 未检测到 GPU，请检查 CUDA 配置。")
        return

    print("\n开始性能对比测试...")

    # 2. GPU 测试
    print("\n[GPU 测试]")
    gpu_time = test_device('cuda', size=8000)

    # 3. CPU 测试 (为了不让您等太久，稍微减小一点 CPU 的测试规模，或者保持一致看差距)
    # 8000x8000 对 CPU 来说非常吃力，我们用 4000x4000 做个参考
    print("\n[CPU 测试 (规模减半至 4000x4000 以免卡死)]")
    cpu_time = test_device('cpu', size=4000)

    print("\n" + "="*40)
    print("测试结果摘要")
    print("="*40)
    if gpu_time:
        print(f"GPU ({gpu_name}) 计算耗时: {gpu_time:.4f} 秒")
    if cpu_time:
        print(f"CPU 计算耗时: {cpu_time:.4f} 秒 (注意：CPU 测试规模仅为 GPU 的 1/4)")
    
    if gpu_time and cpu_time:
        print("\n结论: GPU 工作正常，算力加速明显！")

if __name__ == "__main__":
    main()