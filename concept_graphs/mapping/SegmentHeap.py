import heapq
from .Segment import Segment


class SegmentHeap:
    def __init__(self, max_size: int = 9):
        self.max_size = max_size
        self.heap = []

    def __iter__(self):
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return f"SegmentHeap of size {len(self)} with max size {self.max_size} and segment scores {[s.score for s in self]}"

    def push(self, segment: Segment):
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, segment)
        else:
            heapq.heappushpop(self.heap, segment)

    def extend(self, other: "SegmentHeap"):
        for segment in other:
            self.push(segment)
