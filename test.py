import ray
import numpy as np 
# 初始化Ray
ray.init()



# 定义远程Worker
@ray.remote
class Worker:
    def __init__(self, shared_data):
        self.shared_data = shared_data

    def collect_data(self):
        # 收集transition数据
        transition = [np.random.random(size = (10,3)) for _ in range(np.random.randint(0,10))]

        # 调用远程处理函数对transition数据进行处理
        processed_data = process_data.remote(transition)

        # 将处理后的数据添加到共享内存
        current_data = ray.get(self.shared_data)
        current_data.append(processed_data)
        # self.shared_data = ray.put(current_data,self.shared_data)
        ray.put(current_data,self.shared_data)
        return processed_data

# 定义远程处理函数
@ray.remote
def process_data(transition):
    # 对transition数据进行处理
    # ...

    # 返回处理后的数据
    return transition

# 定义Learner
@ray.remote
class Learner:
    def __init__(self, shared_data):
        self.shared_data = shared_data

    def train(self):
        # 周期性访问共享内存来获取训练数据
        current_data = ray.get(self.shared_data)
        print("########",current_data)
        # 处理所有的收集到的数据
        for i,data in enumerate(current_data):
            print(f"Learner Check: {i}", data)
            # 执行训练逻辑，使用processed_data进行训练
            # ...

        # 清空共享内存
        self.shared_data = ray.put([])

        return True
    
# 创建共享内存对象
shared_data = ray.put([])  # 初始化为空列表
# 创建5个远程Worker
workers = [Worker.remote(shared_data) for _ in range(5)]

# 创建Learner
learner = Learner.remote(shared_data)

# 并行收集数据
tasks = [worker.collect_data.remote() for worker in workers]
learner_task = [learner.train.remote()]

while len(tasks) > 0:
    # 等待所有Worker完成数据收集
    ready_task,tasks = ray.wait(tasks, num_returns=5)
    # 等待Learner完成训练
    print(len(ready_task),len(tasks))
