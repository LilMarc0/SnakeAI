class Snake:
    '''
    length - snake's body length in pixels
    x & y  - head coordinates
    '''
    def __init__(self, length, x, y, boundX, boundY):
        self.length = length
        self.x = x
        self.y = y
        self.body = [[self.x, self.y-i] for i in range(self.length)]
        self.direction = 2
        self.boundX = boundX
        self.boundY = boundY
        self.head = self.body[0]
        self.wrongDirection = False
        assert(self.length == len(self.body))

    @property
    def snakeBody(self):
        return self.body

    '''
    Direction changes
    '''
    def moveRight(self):
        if self.direction == 0:
            self.wrongDirection = True
            return
        self.direction = 2

    def moveLeft(self):
        if self.direction == 2:
            self.wrongDirection = True
            return
        self.direction = 0

    def moveUp(self):
        if self.direction == 3:
            self.wrongDirection = True
            return
        self.direction = 1

    def moveDown(self):
        if self.direction == 1:
            self.wrongDirection = True
            return
        self.direction = 3

    '''
    Returns 'HIT' if snake hits wall or himself
            'FOOD' if snake eats
            '' otherwise
    '''
    def update(self, food):
        new_head = self.body[0].copy()
        if self.direction == 0:    # left
            new_head[1] -= 1
        elif self.direction == 1:  # up
            new_head[0] -= 1
        elif self.direction == 2:  # right
            new_head[1] += 1
        elif self.direction == 3:  # down
            new_head[0] += 1
        if new_head[0] > self.boundX-1 or new_head[1] > self.boundY-1 or new_head[0] < 0 or new_head[1] < 0\
                or new_head in self.snakeBody:
            return 'HIT'
        if new_head == food:
            self.body.insert(0, new_head)
            self.length += 1
            self.head = new_head.copy()
            return 'FOOD'
        self.head = new_head.copy()
        self.body.insert(0, new_head)
        self.body.pop()
        if self.wrongDirection:
            self.wrongDirection = False
            return 'WRONG'
        else:
            return 'MOVED'
