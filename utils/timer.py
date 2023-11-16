"""This File Contains Some Useful Tools"""


import time


class DynamicCounter:
    def __init__(self, total, str_before, interval):
        self._count = 0
        self.total = total
        self.str_before = str_before
        self.interval = interval
        self.start_time = time.time()  # 记录开始时间
        self.last_display_time = time.time()  # 上次显示时间
        self.display_progress()

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value
        self.display_progress()

    def increment(self):
        self._count += 1
        if self._count % self.interval == 0:
            self.display_progress()
        elif self._count == self.total:
            self.display_progress()
            print()

    def stop(self):
        self._count = self.total
        self.display_progress()
        print()

    def display_progress(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time  # 从计数器创建到现在经过的时间
        time_since_last_display = current_time - self.last_display_time  # 上次显示到现在经过的时间
        self.last_display_time = current_time  # 更新上次显示时间
        # 估算次数增长速度
        estimated_growth_rate = self._count / elapsed_time if elapsed_time else 0
        percentage = (self._count / self.total) * 100 if self.total else 0
        estimated_total_time = (self.total - self._count) / estimated_growth_rate if estimated_growth_rate else 0
        print(
            f"\r{self.str_before} - {percentage:.2f}% - {self._count} / {self.total} - "
            f"Time Elapsed: {elapsed_time:.2f}s - Estimated Growth Rate: {estimated_growth_rate:.2f}/s - "
            f"Estimated Time to Complete: {estimated_total_time:.2f}s",
            end="", flush=True)

    def estimate_total_time(self):
        if self._count == 0:
            return "No data to estimate."
        return (self.total - self._count) / (self._count / (time.time() - self.start_time))


if __name__ == "__main__":
    counter = DynamicCounter(0, " ", 10)
