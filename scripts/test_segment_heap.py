from concept_graphs.mapping.Segment import Segment
from concept_graphs.mapping.SegmentHeap import SegmentHeap
import numpy as np

scores = np.arange(100)
np.random.shuffle(scores)

vheap = SegmentHeap(max_size=9)
for i in scores:
    vheap.push(Segment(None, None, None, i, None))

print(vheap)
