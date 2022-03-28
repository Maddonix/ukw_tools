from faker import Faker

class DataGenTests:
    def __init__(self):
        self.factory = Faker()

    def text(self):
        return self.factory.text()