import concurrent.futures
import threading
import unittest
import numpy as np

from .batch_policy import compare, noises
from .batch_service import Aggregator


class BatchTests(unittest.TestCase):
    def test_noise_is_independent_of_batch_membership_order(self):
        records = [{'seed': seed} for seed in (77, 13, 99, 2)]
        batch = noises(records)
        for i, record in enumerate(records):
            np.testing.assert_array_equal(batch[i], noises([record])[0])
        np.testing.assert_array_equal(noises(records[::-1]), batch[::-1])

    def test_compare_rejects_numerical_drift(self):
        self.assertFalse(compare(np.zeros((50, 7)), np.ones((50, 7)))['passed'])
        self.assertTrue(compare(np.ones((50, 7)), np.ones((50, 7)))['passed'])

    def test_aggregation_preserves_each_request(self):
        class Fake:
            def responses(self, records):
                return [{'seed': r['seed'], 'size': len(records)} for r in records]
        aggregator = Aggregator(Fake(), {}, maximum=4, wait_ms=100)
        threading.Thread(target=aggregator.run, daemon=True).start()
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda i: aggregator.submit({'op': 'predict', 'seed': i}), range(4)))
        self.assertEqual([r['seed'] for r in results], list(range(4)))
        self.assertEqual([r['size'] for r in results], [4]*4)
        self.assertEqual(aggregator.actual_batches, [4])

    def test_zero_wait_batches_already_queued_requests(self):
        class Fake:
            def responses(self, records):
                return [{'seed': r['seed'], 'size': len(records)} for r in records]
        aggregator = Aggregator(Fake(), {}, maximum=2, wait_ms=0)
        jobs=[]
        for index in range(4):
            import time
            job=({'op':'predict','seed':index},threading.Event(),{},time.perf_counter())
            jobs.append(job)
            aggregator.jobs.put(job)
        threading.Thread(target=aggregator.run,daemon=True).start()
        for job in jobs:
            self.assertTrue(job[1].wait(1))
        self.assertEqual([job[2]['size'] for job in jobs],[2]*4)
        self.assertEqual(aggregator.actual_batches,[2,2])


if __name__ == '__main__':
    unittest.main()
