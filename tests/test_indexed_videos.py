import json
from pathlib import Path

import pytest

from action_semantics.indexed_videos import (
    _as_string_list,
    _parse_inventory_items,
    flatten_indexed_videos,
    prepare_indexed_videos,
)
from action_semantics.models import ActionTriple, ClipRecord
from action_semantics.month1 import add_record_inventories
from action_semantics.text import clip_text_segments


def test_flatten_indexed_videos_preserves_real_clip_fields(tmp_path):
    source_path = tmp_path / "indexed-videos.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "video_id": 42,
                "youtube_id": "abc123",
                "source": "Youtube",
                "url": "https://example.test/watch?v=abc123",
                "title": "Repair a chair",
                "summary": "A complete chair repair.",
                "goal": "Make the chair safe again.",
                "views": 1200,
                "likes": 75,
                "comment_count": 4,
                "subscribers": 900,
                "clip_count": 1,
                "category": {"id": 11, "name": "Furniture & Decor"},
                "clips": [
                    {
                        "name": "Tighten the loose chair leg",
                        "description": "Use a screwdriver to tighten the leg screw.",
                        "goal": "Stop the chair from wobbling.",
                        "tools": "Screwdriver, clamp",
                        "supplies": "Wood glue",
                        "start": 3.0,
                        "end": 21.5,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    clips, profile = flatten_indexed_videos(source_path)

    assert clips[0]["clip_id"] == "indexed-video-42-segment-3-21p5"
    assert clips[0]["title"] == "Tighten the loose chair leg"
    assert clips[0]["gemini_metadata"]["clip"]["tools"] == ["Screwdriver", "clamp"]
    assert clips[0]["gemini_metadata"]["clip"]["supplies"] == ["Wood glue"]
    source_video = clips[0]["gemini_metadata"]["source_video"]
    assert source_video == {
        "video_id": "42",
        "youtube_id": "abc123",
        "source": "Youtube",
        "url": "https://example.test/watch?v=abc123",
        "category": {"id": 11, "name": "Furniture & Decor"},
        "title": "Repair a chair",
        "summary": "A complete chair repair.",
        "goal": "Make the chair safe again.",
        "views": 1200,
        "likes": 75,
        "comment_count": 4,
        "subscribers": 900,
        "declared_clip_count": 1,
    }
    assert profile["video_count"] == 1
    assert profile["clip_count"] == 1
    assert profile["raw_clip_count"] == 1
    assert profile["rejected_clip_count"] == 0
    assert profile["month1_month2_ready"] is True
    assert profile["structured_search_ready"] is True
    assert profile["comparative_evaluation_ready"] is False
    assert profile["month3_ready"] is False
    assert len(profile["month3_blockers"]) == 3


def test_prepare_indexed_videos_writes_flat_export_and_profile(tmp_path):
    source_path = tmp_path / "indexed-videos.jsonl"
    source_path.write_text(
        '{"video_id": 1, "clips": '
        '[{"name": "Clean a pipe", "start": 1, "end": 2}]}\n',
        encoding="utf-8",
    )

    paths = prepare_indexed_videos(source_path, tmp_path / "prepared")

    assert paths["clips"].exists()
    assert paths["rejected"].exists()
    assert paths["profile"].exists()
    assert json.loads(paths["clips"].read_text(encoding="utf-8").strip())["clip_id"] == (
        "indexed-video-1-segment-1-2"
    )
    assert paths["rejected"].read_text(encoding="utf-8") == ""


def test_duplicate_segments_are_merged_and_invalid_intervals_are_rejected(tmp_path):
    source_path = tmp_path / "indexed-videos.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "video_id": 7,
                "clip_count": 4,
                "clips": [
                    {
                        "name": "Mount the thermostat",
                        "description": "Use the marked holes.",
                        "tools": "Screwdriver",
                        "start": 10.0,
                        "end": 20.0,
                    },
                    {
                        "name": "Secure thermostat to wall",
                        "description": "Use the marked holes.",
                        "tools": "Screwdriver used for Tightening screws.",
                        "supplies": "Wall anchors",
                        "start": 10.000,
                        "end": 20.000,
                    },
                    {
                        "name": "Mount the thermostat",
                        "description": "Use the marked holes.",
                        "tools": "Screwdriver",
                        "start": 10,
                        "end": 20,
                    },
                    {
                        "name": "Impossible segment",
                        "start": 30,
                        "end": 30,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    clips, profile = flatten_indexed_videos(source_path)

    assert len(clips) == 1
    assert clips[0]["clip_id"] == "indexed-video-7-segment-10-20"
    assert clips[0]["title"] == "Mount the thermostat"
    metadata = clips[0]["gemini_metadata"]["clip"]
    assert metadata["aliases"] == ["Secure thermostat to wall"]
    assert metadata["variant_count"] == 3
    assert metadata["source_clip_indices"] == [0, 1, 2]
    assert metadata["tools"] == ["Screwdriver"]
    assert metadata["supplies"] == ["Wall anchors"]
    assert len(metadata["source_variants"]) == 3
    assert profile["raw_clip_count"] == 4
    assert profile["valid_source_clip_count"] == 3
    assert profile["canonical_clip_count"] == 1
    assert profile["rejected_clip_count"] == 1
    assert profile["rejection_reason_counts"] == {"non_positive_duration": 1}
    assert profile["duplicate_segment_group_count"] == 1
    assert profile["merged_duplicate_row_count"] == 2
    assert profile["alias_segment_group_count"] == 1
    assert profile["alias_count"] == 1

    paths = prepare_indexed_videos(source_path, tmp_path / "prepared")
    rejected = json.loads(paths["rejected"].read_text(encoding="utf-8").strip())
    assert rejected["reason"] == "non_positive_duration"
    assert rejected["name"] == "Impossible segment"


def test_source_models_reject_unrecognized_fields(tmp_path):
    source_path = tmp_path / "indexed-videos.jsonl"
    source_path.write_text(
        '{"video_id": 1, "unexpected": true, "clips": '
        '[{"name": "Clean a pipe", "start": 1, "end": 2}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra_forbidden"):
        flatten_indexed_videos(source_path)


_PRIVATE_SAMPLE = Path(__file__).parents[1] / "indexed-videos-250.jsonl"


@pytest.mark.skipif(not _PRIVATE_SAMPLE.exists(), reason="private sample is not checked in")
def test_private_sample_canonicalization_audit_counts():
    clips, profile = flatten_indexed_videos(_PRIVATE_SAMPLE)

    assert len(clips) == 1663
    assert profile["video_count"] == 250
    assert profile["raw_clip_count"] == 1703
    assert profile["valid_source_clip_count"] == 1700
    assert profile["canonical_clip_count"] == 1663
    assert profile["rejected_clip_count"] == 3
    assert profile["duplicate_segment_group_count"] == 36
    assert profile["merged_duplicate_row_count"] == 37
    assert profile["exact_duplicate_segment_group_count"] == 13
    assert profile["metadata_variant_segment_group_count"] == 23
    assert profile["alias_segment_group_count"] == 2
    assert profile["alias_count"] == 3


def test_indexed_video_inventory_metadata_is_not_parsed_as_action_text():
    clip = ClipRecord(
        clip_id="indexed-video-1-clip-000",
        title="Tighten the hinge",
        description="Drive the screw into the hinge.",
        gemini_metadata={
            "source_video": {"category": "Cleaning", "title": "How to use a drill"},
            "clip": {
                "tools": ["Scrubbing brush"],
                "supplies": ["Baking soda"],
                "aliases": ["Secure the hinge"],
                "description_variants": ["Drive the screw into the hinge."],
                "source_variants": [{"name": "Tighten the hinge"}],
            },
        },
    )

    segments = clip_text_segments(clip)

    assert [(segment.source_field, segment.text) for segment in segments] == [
        ("title", "Tighten the hinge"),
        ("description", "Drive the screw into the hinge."),
    ]


def test_record_tools_are_preserved_as_separate_context():
    clip = ClipRecord(
        clip_id="clip-1",
        gemini_metadata={
            "clip": {"tools": ["Cordless Drill"], "supplies": ["Wood glue"]}
        },
    )
    triple = ActionTriple(
        record_type="clip",
        record_id="clip-1",
        source_field="title",
        action="drill",
        action_lemma="drill",
        action_text="Drill",
        sentence="Drill the pilot hole.",
        extraction_method="test",
    )

    enriched = add_record_inventories([triple], [clip], [])

    assert enriched[0].tool_lemmas == []
    assert enriched[0].context_tool_lemmas == ["cordless", "cordless drill", "drill"]
    assert enriched[0].context_material_lemmas == ["glue", "wood", "wood glue"]


def test_annotated_tool_inventory_keeps_primary_names_only():
    value = (
        "Cordless Drill Ridgid alternatives: Impact driver used for Driving screws., "
        "Work Gloves Unknown used for Protecting hands."
    )

    assert _as_string_list(value) == ["Cordless Drill Ridgid", "Work Gloves"]


def test_annotated_inventory_preserves_alternatives_purpose_and_raw_text():
    value = (
        "Rake Unknown alternatives: Leaf blower, Shovel used for Clearing debris., "
        "String Unknown used for Marking the path."
    )

    items = _parse_inventory_items(value)

    assert [item.name for item in items] == ["Rake", "String"]
    assert items[0].alternatives == ["Leaf blower", "Shovel"]
    assert items[0].purpose == "Clearing debris"
    assert items[0].raw.startswith("Rake Unknown alternatives:")
    assert items[1].alternatives == []
    assert items[1].purpose == "Marking the path"
