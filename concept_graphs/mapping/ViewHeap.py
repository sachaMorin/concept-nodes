import heapq
from .View import View


class ViewHeap:
    def __init__(self, max_size: int = 9):
        self.max_size = max_size
        self.heap = []

    def __iter__(self):
        return iter(self.heap)

    def __repr__(self):
        return f"ViewHeap of size {len(self.heap)} with max size {self.max_size} and view scores {[view.score for view in self]}"

    def push(self, view: View):
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, view)
        else:
            heapq.heappushpop(self.heap, view)

    def extend(self, other: "ViewHeap"):
        for view in other:
            self.push(view)

