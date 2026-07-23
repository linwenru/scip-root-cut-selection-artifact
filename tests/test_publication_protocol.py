import unittest
from pathlib import Path

from scip_cut_trace_v2.publication_protocol import (
    CandidateInstance,
    _schedule_position_counts,
    build_balanced_arm_schedule,
    select_completion_enriched,
    select_distinct_groups,
)


def _candidate(index, group=None):
    return CandidateInstance(
        instance_id=f"instance-{index:03d}",
        instance_name=f"instance-{index:03d}.mps.gz",
        path=Path(f"instance-{index:03d}.mps.gz"),
        official_group=group if group is not None else f"group-{index:03d}",
        evaluation_stratum="training_grouped",
        source_trace_elapsed_seconds=float(index),
        policy_eligible_decisions=1,
        policy_eligible_candidates=10,
        policy_eligible_applied_labels=2,
    )


class PublicationProtocolTest(unittest.TestCase):
    def test_distinct_group_selection_is_deterministic(self):
        candidates = [
            _candidate(0, "shared"),
            _candidate(1, "shared"),
            _candidate(2, "second"),
        ]

        selected = select_distinct_groups(candidates, 2)

        self.assertEqual([item.instance_id for item in selected], ["instance-000", "instance-002"])

    def test_active_selection_has_completion_and_hardness_quotas(self):
        candidates = [_candidate(index) for index in range(120)]

        selected, strata = select_completion_enriched(candidates, 40)

        self.assertEqual(len(selected), 40)
        self.assertEqual([item["selected_instances"] for item in strata], [30, 10])
        self.assertEqual(len({item.group_key for item in selected}), 40)
        self.assertEqual(
            [item.instance_id for item in selected[:30]],
            [f"instance-{index:03d}" for index in range(30)],
        )
        self.assertEqual(len(selected[30:]), 10)
        self.assertGreaterEqual(int(selected[30].instance_id[-3:]), 30)

    def test_balanced_schedule_places_every_arm_thirty_times_per_position(self):
        instances = [
            {"instance_id": f"instance-{index:02d}"} for index in range(40)
        ]
        schedule = build_balanced_arm_schedule(
            instances,
            [0, 1, 2],
            ["random-rank", "efficacy-rank", "adaptive-score"],
            "frozen-key",
        )

        self.assertEqual(len(schedule), 120)
        counts = _schedule_position_counts(schedule)
        self.assertTrue(
            all(count == 30 for positions in counts.values() for count in positions.values())
        )
        self.assertEqual(
            len({(block["instance_id"], block["seed"]) for block in schedule}),
            120,
        )


if __name__ == "__main__":
    unittest.main()
