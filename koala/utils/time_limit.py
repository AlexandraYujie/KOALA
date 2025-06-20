import time
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls  # 最大调用次数，例如1200
        self.period = period        # 时间窗口长度（秒），例如60
        self.timestamps = deque()   # 存储请求时间戳的队列
        self.lock = threading.Lock()  # 确保线程安全

    def acquire(self):
        """阻塞直到获得一个调用名额"""
        while True:
            with self.lock:
                now = time.time()
                # 移除超过时间窗口的旧时间戳
                while self.timestamps and self.timestamps[0] < now - self.period:
                    self.timestamps.popleft()
                # 检查是否允许新请求
                if len(self.timestamps) < self.max_calls:
                    self.timestamps.append(now)
                    return  # 成功获取名额，退出循环

            # 计算需要等待的时间并休眠
            oldest = self.timestamps[0] if self.timestamps else now
            wait_time = oldest + self.period - now
            if wait_time > 0:
                time.sleep(wait_time)



    # 使用示例
if __name__ == '__main__':
    import numpy as np

    def call_api(limiter):
        limiter.acquire()
        x = 1
        std_dev = 0.5
        while True:
            random_number = np.random.normal(loc=x, scale=std_dev)
            if random_number > 0:
                break
        print(random_number)
        time.sleep(random_number)

    # 模拟多线程调用
    def worker(limiter, progress):
        result_list = []
        sample_in_task = 60
        for i in range(sample_in_task):  # 每个线程发起2000次调用
            result = f"task: {progress} | {i+1}/{sample_in_task}"
            call_api(limiter)
            result_list.append(result)
            print(result)
        return result_list

    def get_expansion():
        limiter = RateLimiter(60, 60)
        max_threads = 2
        print(">>> max_threads:", max_threads)
        total_task_num = 2
        total_result_list = []
        with ThreadPoolExecutor(max_threads) as executor:
            futures = [executor.submit(worker, limiter, progress=f"{i + 1}/{total_task_num}") for i in range(total_task_num)]
            for future in as_completed(futures):
                result_list = future.result()
                total_result_list.extend(result_list)
        print("length of total result list:", len(total_result_list))
        print(total_result_list)



