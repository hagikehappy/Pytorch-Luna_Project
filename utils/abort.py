"""This File Contains Some Self-Defined Abort Function"""


class SettingsAbort(Exception):
    """用于settings中未找到匹配项时使用的Error"""
    def __init__(self, message="Settings Abort!!!"):
        self.message = message

    def __str__(self):
        return self.message


class CacheAbort(Exception):
    """用于Cache中未找到匹配项时使用的Error"""
    def __init__(self, message="Cache Abort!!!"):
        self.message = message

    def __str__(self):
        return self.message


class TrainAbort(Exception):
    """用于在训练发生错误时使用的Error"""
    def __init__(self, message="Train Abort!!!"):
        self.message = message

    def __str__(self):
        return self.message


class PredictAbort(Exception):
    """用于在训练发生错误时使用的Error"""
    def __init__(self, message="Predict Abort!!!"):
        self.message = message

    def __str__(self):
        return self.message

