"""This File Contains Some Self-Defined Abort Function"""


class SettingsAbort(Exception):
    """用于settings中未找到匹配项时使用的Error"""
    def __init__(self, message="Settings Abort!!!"):
        self.message = message

    def __str__(self):
        return self.message
