from faker import Faker

class TestDataGen:
    def __init__(self):
        self.factory = Faker()

    def text(self):
        return self.factory.text()