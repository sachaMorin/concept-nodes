import heapq
from .View import View


class ViewHeap:
    def __init__(self, max_size: int = 9):
        self.max_size = max_size
        self.heap = []

    def __iter__(self):
        return iter(self.heap)

    def push(self, view: View):
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, view)
        else:
            heapq.heappushpop(self.heap, view)

    def extend(self, other: "ViewHeap"):
        for view in other:
            self.push(view)

