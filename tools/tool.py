"""This File Contains Some Useful Tools"""


import time


class DynamicCounter:
    def __init__(self, total, str_before):
        self.count = 0
        self.total = total
        self.str_before = str_before

    def increment(self):
        self.count += 1
        self.display()

    def display(self):
        percentage = (self.count / self.total) * 100
        print(f"\r{self.str_before} : {percentage:.2f}% : {self.count} / {self.total}", end="", flush=True)
