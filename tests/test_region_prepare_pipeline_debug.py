"""DEBUG scheduler probes; synchronization events, no timing-speed assertions."""
import threading
import unittest
from l0_regions.preparation_runtime import prefetched


class PreparePipeline(unittest.TestCase):
    def test_next_load_overlaps_consumer_without_changing_order(self):
        reached=threading.Event();seen=[]
        def load(i):
            seen.append(i)
            if i==1:reached.set()
            return i*2
        with prefetched([0,1,2],load) as batches:
            first=next(batches)
            self.assertEqual(first[:2],(0,0))
            self.assertTrue(reached.wait(3),'Next load must start while caller holds current result')
            self.assertEqual(seen,[0,1],'Must not load the whole cohort ahead')
            rest=list(batches)
        self.assertEqual([r[:2] for r in rest],[(1,2),(2,4)])
        self.assertEqual(seen,[0,1,2])

    def test_cpu_exception_reaches_consumer(self):
        def load(i):
            if i==1:raise ValueError('changed record')
            return i
        with prefetched([0,1],load) as batches:
            self.assertEqual(next(batches)[1],0)
            with self.assertRaisesRegex(ValueError,'changed record'):next(batches)

    def test_close_does_not_process_unscheduled_records(self):
        seen=[]
        with prefetched(range(20),lambda i:seen.append(i)) as batches:
            next(batches)
        self.assertTrue(set(seen)<={0,1})

    def test_empty_and_tail_coverage(self):
        with prefetched([],lambda _:self.fail()) as batches:self.assertEqual(list(batches),[])
        requests=[[0,1,2],[3,4,5],[6]]
        with prefetched(requests,lambda ids:ids.copy()) as batches:
            self.assertEqual([v for _,v,_,_ in batches],requests)


if __name__=='__main__':unittest.main()
