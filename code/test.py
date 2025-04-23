#just test
import pickle

# 替换为实际的 .pkl 文件路径
#"C:\Users\zhang\Desktop\temp_cache_100.pkl"
file_path = "C:/Users/zhang/Desktop/temp_cache_100.pkl"

# 打开文件并加载内容
with open(file_path, 'rb') as f:
    try:
        data = pickle.load(f)  # 尝试直接加载
    except UnicodeDecodeError:
        # 若出现编码错误（如国外代码生成的文件），尝试指定编码
        data = pickle.load(f, encoding='latin1')

if isinstance(data, dict):
    print("数据为字典类型，前 50 个键值对内容：")
    count = 0
    for key, value in data.items():
        print(f"键: {key}, 值: {value}")
        count += 1
        if count == 300:
            break
else:
    print(f"数据不是字典类型。数据类型为 {type(data)}")