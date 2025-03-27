import json

import pytest

from reasoning_gym.algorithmic.graph_color import GraphColorConfig, GraphColorCurriculum, GraphColorDataset
from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode


def test_graph_color():
    """Test basic properties and solution of generated items"""
    config = GraphColorConfig(
        seed=42,
        size=10,
        min_num_vertices=10,
        max_num_vertices=10,
        num_colors=4,
        edge_probability=0.4,
    )
    dataset = GraphColorDataset(config)

    # easy
    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Test the scoring
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # medium
    config = GraphColorConfig(
        seed=42,
        size=1,
        min_num_vertices=10,
        max_num_vertices=10,
        num_colors=3,
        edge_probability=0.1,
    )
    dataset = GraphColorDataset(config)

    for item in dataset:
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # hard
    config = GraphColorConfig(
        seed=42,
        size=1,
        min_num_vertices=15,
        max_num_vertices=15,
        num_colors=3,
        edge_probability=0.1,
    )
    dataset = GraphColorDataset(config)

    for item in dataset:
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0

    # v hard
    config = GraphColorConfig(
        seed=42,
        size=1,
        min_num_vertices=50,
        max_num_vertices=50,
        num_colors=3,
        edge_probability=0.1,
    )
    dataset = GraphColorDataset(config)

    for item in dataset:
        assert dataset.score_answer(answer=json.dumps(item["metadata"]["possible_answer"]), entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0


def test_graph_color_curriculum():
    curriculum = GraphColorCurriculum()

    base_value = {"size": 150, "seed": 1}

    context = DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND)
    base_cfg: GraphColorConfig = curriculum.generate_configuration(base_value, context=context)
    assert base_cfg.size == 150
    assert base_cfg.seed == 1
    assert base_cfg.min_num_vertices == base_cfg.max_num_vertices == 10
    assert base_cfg.num_colors == base_cfg.num_colors == 5

    curriculum.increment_attr_level("num_vertices")
    cfg = curriculum.generate_configuration(base_value, context=context)
    assert cfg.min_num_vertices == 20

    curriculum.increment_attr_level("num_colors")
    cfg = curriculum.generate_configuration(base_value)
    assert cfg.num_colors == 4
    curriculum.increment_attr_level("num_colors")
    cfg = curriculum.generate_configuration(base_value)
    assert cfg.num_colors == 3
