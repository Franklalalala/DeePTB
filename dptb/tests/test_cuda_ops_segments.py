import torch


def test_repeated_segment_layout_sorts_repeated_routes_and_builds_cpu_ptr():
    from dptb.nn.cuda_ops.segments import repeated_segment_layout

    graph_index = torch.tensor([2, 0, 1], dtype=torch.long)

    layout = repeated_segment_layout(
        graph_index,
        3,
        repeat=2,
        cache_name="test_segments_repeat",
    )

    assert layout.order is not None
    assert layout.unorder is not None
    assert layout.sorted_index.tolist() == [0, 0, 1, 1, 2, 2]
    assert layout.ptr_cpu.device.type == "cpu"
    assert layout.ptr_cpu.tolist() == [0, 2, 4, 6]

    restored = layout.sorted_index.index_select(0, layout.unorder)
    assert restored.tolist() == [2, 2, 0, 0, 1, 1]


def test_repeated_segment_layout_uses_cache_for_same_tensor_version():
    from dptb.nn.cuda_ops.segments import repeated_segment_layout

    graph_index = torch.tensor([0, 1, 1, 2], dtype=torch.long)

    first = repeated_segment_layout(
        graph_index,
        3,
        repeat=1,
        assume_sorted=True,
        cache_name="test_segments_cache",
    )
    second = repeated_segment_layout(
        graph_index,
        3,
        repeat=1,
        assume_sorted=True,
        cache_name="test_segments_cache",
    )

    assert second is first
    assert first.order is None
    assert first.unorder is None
    assert first.sorted_index.tolist() == [0, 1, 1, 2]
    assert first.ptr_cpu.tolist() == [0, 1, 3, 4]
