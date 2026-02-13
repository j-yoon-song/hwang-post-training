from synth_parallel.bucketing import BalancedBucketSampler, assign_bucket


def test_assign_bucket():
    boundaries = [0, 10, 20, 40]
    assert assign_bucket(5, boundaries) == 0
    assert assign_bucket(10, boundaries) == 1
    assert assign_bucket(35, boundaries) == 2
    assert assign_bucket(999, boundaries) == 2


def test_balanced_sampler_quota_and_fill():
    boundaries = [0, 10, 20, 30]
    sampler = BalancedBucketSampler(boundaries, sample_size=6, oversample_factor=2.0, seed=7)

    # Bucket 0: many samples
    for i in range(20):
        sampler.add({"id": f"a{i}", "length_approx": 5}, 5)

    # Bucket 1: sparse
    for i in range(2):
        sampler.add({"id": f"b{i}", "length_approx": 15}, 15)

    # Bucket 2: many
    for i in range(20):
        sampler.add({"id": f"c{i}", "length_approx": 25}, 25)

    selected, stats = sampler.finalize()

    assert len(selected) == 6
    assert sum(s.kept for s in stats.values()) <= 6


def test_balanced_sampler_can_fill_target_true_with_extras():
    boundaries = [0, 10, 20]
    sampler = BalancedBucketSampler(boundaries, sample_size=4, oversample_factor=2.0, seed=7)

    for i in range(20):
        sampler.add({"id": f"a{i}", "length_approx": 5}, 5)

    assert sampler.buffered_count() == 4
    assert sampler.can_fill_target() is True


def test_balanced_sampler_can_fill_target_false_when_skewed_and_small_capacity():
    boundaries = [0, 10, 20, 30]
    sampler = BalancedBucketSampler(boundaries, sample_size=6, oversample_factor=2.0, seed=7)

    for i in range(100):
        sampler.add({"id": f"a{i}", "length_approx": 5}, 5)

    # Bucket0 quota=2, capacity=4, so with only one bucket populated
    # the sampler cannot fill all 6 targets.
    assert sampler.buffered_count() == 4
    assert sampler.can_fill_target() is False
