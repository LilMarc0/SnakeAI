class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def position(self):
        return [self.x, self.y]

    def move(self, x, y):
        self.x = x
        self.y = y
