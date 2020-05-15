class Stack:
    def __init__(self):
        self.items = []

    def push(self, value):
        self.items.append(value)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return self.size() == 0

    def size(self):
        return len(self.items)

    def peek(self):
        return self.items[self.size() - 1]

    def peek_peek(self):
        return self.items[self.size() - 2]

