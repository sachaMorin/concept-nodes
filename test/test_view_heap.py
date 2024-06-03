from concept_graphs.mapping.View import View
from concept_graphs.mapping.ViewHeap import ViewHeap
import numpy as np

scores = np.arange(100)
np.random.shuffle(scores)

vheap = ViewHeap(max_size=9)
for i in scores:
    vheap.push(View(None, None, None, i, None))

print(vheap)
