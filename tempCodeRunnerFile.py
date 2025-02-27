def block_sort(arr):
    max_val = max(arr)
    size = len(arr)
    block_size = int(np.sqrt(size))
    blocks = [[] for _ in range((max_val // block_size) + 1)]
    
    for num in arr:
        blocks[num // block_size].append(num)
    
    for block in blocks:
        block.sort()
    
    sorted_arr = []
    for block in blocks:
        sorted_arr.extend(block)
    
    return sorted_arr