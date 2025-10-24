import math

def calculate_mpo_shape(in_features, out_features=None):
    """
    根据输入特征数自动计算 MPO 的输入和输出形状
    
    Args:
        in_features (int): 输入特征数
        out_features (int, optional): 输出特征数，如果为 None 则与输入相同
    
    Returns:
        tuple: (mpo_input_shape, mpo_output_shape)
    
    Examples:
        >>> calculate_mpo_shape(1024)
        ([32, 32], [32, 32])
        >>> calculate_mpo_shape(1024, 512)
        ([32, 32], [16, 32])
        >>> calculate_mpo_shape(768)
        ([28, 28], [28, 28])
    """
    
    if out_features is None:
        out_features = in_features
    
    def find_best_factors(n, target_dim=2):
        """
        找到最接近平方根的两个因子
        """
        if n <= 0:
            return [1] * target_dim
        
        # 如果是完全平方数
        sqrt_n = int(math.sqrt(n))
        if sqrt_n * sqrt_n == n:
            return [sqrt_n] * target_dim
        
        # 找到最接近平方根的两个因子
        best_factors = None
        min_diff = float('inf')
        
        # 尝试不同的因子组合
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                j = n // i
                # 计算与平方根的差异
                diff = abs(i - sqrt_n) + abs(j - sqrt_n)
                if diff < min_diff:
                    min_diff = diff
                    best_factors = [i, j]
        
        # 如果找不到合适的因子，使用最接近的平方数
        if best_factors is None:
            sqrt_n = int(math.sqrt(n))
            return [sqrt_n, sqrt_n]
        
        return sorted(best_factors)
    
    # 计算输入形状
    in_shape = find_best_factors(in_features)
    
    # 计算输出形状
    out_shape = find_best_factors(out_features)
    
    return in_shape, out_shape

