"""
Relevant links used to create this test
1. https://heidloff.net/article/efficient-fine-tuning-lora/
2.
"""
from peft import LORETTaConfig


class TestLORETTA:
    @staticmethod
    def config_test():
        loretta_config = LORETTaConfig()

if __name__=="__main__":
    TestLORETTA.config_test()
    pass