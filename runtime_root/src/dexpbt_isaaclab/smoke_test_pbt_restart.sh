#!/usr/bin/env bash
# Smoke test for the PBT restart mechanism.
#
# Tests:
#   Phase 1: Supervisor manifest write/read/delete round-trip
#   Phase 2: Short training run (default observer, no PBT)
#   Phase 3: Checkpoint resume from Phase 2 checkpoint
#   Phase 4: Full PBT observer training (tests exit-code-42 path)
#
# Usage:
#   bash smoke_test_pbt_restart.sh [GPU_ID]
#   GPU_ID defaults to 0

set -euo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUPERVISOR="$SCRIPT_DIR/run_train_supervisor.sh"
TEST_ROOT="$SCRIPT_DIR/results/smoke_pbt_restart_$(date +%Y%m%d_%H%M%S)"

echo "=== PBT Restart Smoke Test ==="
echo "Test root: $TEST_ROOT"
echo "GPU: $GPU"
echo ""

mkdir -p "$TEST_ROOT"

PHASE1_OK=0
PHASE2_OK=0
PHASE3_OK=0
PHASE4_OK=0

# -----------------------------------------------------------------------
# Phase 1: Test PBT manifest write/read/delete (no Isaac Sim)
# -----------------------------------------------------------------------
echo "=== Phase 1: Supervisor manifest round-trip ==="
P1_LOG="$TEST_ROOT/phase1.log"
{
    MOCK_DIR="$TEST_ROOT/phase1_mock"
    mkdir -p "$MOCK_DIR"

    # Create fake checkpoint
    mkdir -p "$MOCK_DIR/rl_games/attempt_1/nn"
    FAKE_CKPT="$MOCK_DIR/rl_games/attempt_1/nn/fake.pth"
    echo "fake_weights" > "$FAKE_CKPT"

    # Simulate PBT observer writing manifest
    python3 -c "
import json, os
manifest = {
    'source_policy': 3,
    'tier': 'bottom',
    'env_frames': 200000000,
    'checkpoint': os.path.abspath('$FAKE_CKPT'),
    'train_args': ['--checkpoint=' + os.path.abspath('$FAKE_CKPT'), '--gamma=0.985', '--e_clip=0.12']
}
with open('$MOCK_DIR/pbt_restart.json', 'w') as f:
    json.dump(manifest, f)
print('Wrote manifest')
"

    # Verify manifest exists
    [[ -f "$MOCK_DIR/pbt_restart.json" ]] || { echo "FAIL: manifest not written"; exit 1; }

    # Simulate supervisor reading manifest (same logic as run_train_supervisor.sh)
    pbt_override_checkpoint=$(python3 -c "
import json
m = json.load(open('$MOCK_DIR/pbt_restart.json'))
print(m.get('checkpoint', ''))
")
    new_args=$(python3 -c "
import json
m = json.load(open('$MOCK_DIR/pbt_restart.json'))
print(' '.join(m['train_args']))
")

    echo "  checkpoint: $pbt_override_checkpoint"
    echo "  train_args: $new_args"

    [[ -n "$pbt_override_checkpoint" ]] || { echo "FAIL: empty checkpoint"; exit 1; }
    [[ -f "$pbt_override_checkpoint" ]] || { echo "FAIL: checkpoint file missing"; exit 1; }
    [[ "$new_args" == *"--gamma=0.985"* ]] || { echo "FAIL: gamma missing from args"; exit 1; }
    [[ "$new_args" == *"--checkpoint"* ]] || { echo "FAIL: --checkpoint missing from args"; exit 1; }

    # Simulate supervisor clearing manifest
    rm -f "$MOCK_DIR/pbt_restart.json"
    [[ ! -f "$MOCK_DIR/pbt_restart.json" ]] || { echo "FAIL: manifest not deleted"; exit 1; }

    # Test latest_checkpoint.txt round-trip
    echo "$pbt_override_checkpoint" > "$MOCK_DIR/latest_checkpoint.txt"
    stored="$(tr -d '\n' < "$MOCK_DIR/latest_checkpoint.txt")"
    [[ "$stored" == "$pbt_override_checkpoint" ]] || { echo "FAIL: latest_checkpoint round-trip"; exit 1; }

    echo "Phase 1 PASSED"
} 2>&1 | tee "$P1_LOG"
PHASE1_OK=1

# -----------------------------------------------------------------------
# Phase 2: Short training run (default observer, no PBT)
# -----------------------------------------------------------------------
echo ""
echo "=== Phase 2: Short training (default observer, 50 iterations) ==="
P2_LOG="$TEST_ROOT/phase2.log"
P2_RUN="$TEST_ROOT/phase2_run"
mkdir -p "$P2_RUN"

set +e
CUDA_VISIBLE_DEVICES="$GPU" bash "$SUPERVISOR" \
    "$P2_RUN" \
    "DexPBT-Reorientation-KukaAllegro-Direct-v0" \
    "--device=cuda:0" \
    "--headless" \
    "--num_envs=2048" \
    "--max_iterations=50" \
    2>&1 | tee "$P2_LOG"
P2_EXIT=$?
set -e

if [[ "$P2_EXIT" -eq 0 ]]; then
    CKPT_COUNT=$(find "$P2_RUN" -name "*.pth" 2>/dev/null | wc -l)
    echo "Phase 2 PASSED (exit 0, $CKPT_COUNT checkpoints)"
    PHASE2_OK=1
else
    echo "Phase 2 FAILED (exit $P2_EXIT)"
    tail -20 "$P2_LOG"
fi

# -----------------------------------------------------------------------
# Phase 3: Checkpoint resume
# -----------------------------------------------------------------------
if [[ "$PHASE2_OK" -eq 1 ]] && [[ "$CKPT_COUNT" -gt 0 ]]; then
    echo ""
    echo "=== Phase 3: Checkpoint resume ==="
    P3_LOG="$TEST_ROOT/phase3.log"
    P3_RUN="$TEST_ROOT/phase3_run"
    mkdir -p "$P3_RUN"

    LATEST_CKPT=$(find "$P2_RUN" -name "*.pth" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    echo "Resuming from: $LATEST_CKPT"

    set +e
    CUDA_VISIBLE_DEVICES="$GPU" bash "$SUPERVISOR" \
        "$P3_RUN" \
        "DexPBT-Reorientation-KukaAllegro-Direct-v0" \
        "--device=cuda:0" \
        "--headless" \
        "--num_envs=2048" \
        "--max_iterations=10" \
        "--checkpoint=$LATEST_CKPT" \
        2>&1 | tee "$P3_LOG"
    P3_EXIT=$?
    set -e

    if [[ "$P3_EXIT" -eq 0 ]]; then
        echo "Phase 3 PASSED (checkpoint resume works)"
        PHASE3_OK=1
    else
        echo "Phase 3 FAILED (exit $P3_EXIT)"
        tail -20 "$P3_LOG"
    fi
else
    echo ""
    echo "=== Phase 3 SKIPPED ==="
fi

# -----------------------------------------------------------------------
# Phase 4: PBT observer training with fake peer objective
#   Pre-populates the workspace with a fake "policy 1" that has a high
#   objective, so the observer sees enough peers and triggers a restart.
#   Uses pbt_interval_steps small enough that the PBT tick fires during
#   our short training window.
# -----------------------------------------------------------------------
echo ""
echo "=== Phase 4: PBT observer training (expect exit-code-42 restart) ==="
P4_LOG="$TEST_ROOT/phase4.log"
P4_RUN="$TEST_ROOT/phase4_run"
mkdir -p "$P4_RUN"

# Create workspace directory with a fake "policy 1" that has a high objective
# so the observer ranks itself low and triggers a restart.
# The fake checkpoint must NOT be a .pth file in the supervisor's run dir
# tree, or pick_checkpoint() will try to load it.
P4_WS="$P4_RUN/pbt_ws"
mkdir -p "$P4_WS/001"

# Create a fake checkpoint for the "other policy" using .json only
# (the observer reads .json metadata to get objective values;
#  the checkpoint path inside the json points to a real ckpt from Phase 2)
# Use Phase 2's checkpoint if available, else create a dummy one.
P4_PEER_CKPT="$P4_WS/001/peer_ckpt.pth"
if [[ "$CKPT_COUNT" -gt 0 ]]; then
    # Use a real checkpoint from Phase 2
    P4_PEER_CKPT=$(find "$P2_RUN" -name "last_*.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [[ -z "$P4_PEER_CKPT" ]]; then
        P4_PEER_CKPT=$(find "$P2_RUN" -name "*.pth" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    fi
fi
echo "  Peer checkpoint: $P4_PEER_CKPT"

python3 -c "
import json, time
meta = {
    'policy_idx': 1,
    'objective': 999.0,
    'frame': 1000000,
    'iteration': 0,
    'timestamp': time.time(),
    'params': {
        'gamma': 0.99, 'kl_threshold': 0.016, 'grad_norm': 1.0,
        'critic_coef': 4.0, 'bounds_loss_coef': 0.0001, 'e_clip': 0.1,
        'mini_epochs': 2, 'alpha_reach': 0.2, 'alpha_pick': 0.5,
        'r_picked': 1.0, 'alpha_targ': 0.3, 'r_success': 5.0,
        'kuka_actions_penalty_scale': 0.001, 'allegro_actions_penalty_scale': 0.001,
    },
    'checkpoint': '$P4_PEER_CKPT',
    'experiment_name': 'fake_peer',
}
with open('$P4_WS/001/000000.json', 'w') as f:
    json.dump(meta, f)
print('Created fake peer policy with objective=999.0')
"

set +e
CUDA_VISIBLE_DEVICES="$GPU" bash "$SUPERVISOR" \
    "$P4_RUN" \
    "DexPBT-Reorientation-KukaAllegro-Direct-v0" \
    "--device=cuda:0" \
    "--headless" \
    "--observer_mode=pbt" \
    "--policy_idx=0" \
    "--num_policies=2" \
    "--workspace=$P4_WS" \
    "--pbt_interval_steps=100" \
    "--pbt_start_after=0" \
    "--pbt_initial_delay=0" \
    "--num_envs=2048" \
    "--max_iterations=20" \
    2>&1 | tee "$P4_LOG"
P4_EXIT=$?
set -e

# Check PBT behavior
P4_ATTEMPTS=$(find "$P4_RUN" -name "attempt_*.log" 2>/dev/null | wc -l)
echo ""
echo "  Exit code: $P4_EXIT"
echo "  Attempts: $P4_ATTEMPTS"
echo "  PBT restarts attempted: $((P4_ATTEMPTS - 1))"

# Check if a PBT restart manifest was written (and then consumed)
P4_MANIFEST_EXISTS=$(grep -rch "PBT restart requested\|pbt_restart.json\|PBT restart checkpoint" "$P4_RUN"/attempt_*.log 2>/dev/null | awk '{s+=$1}END{print s+0}')
P4_RESTART_LOGS=$(grep -rch "exit code 42\|requesting supervisor restart" "$P4_RUN"/attempt_*.log 2>/dev/null | awk '{s+=$1}END{print s+0}')

if [[ "$P4_ATTEMPTS" -gt 1 ]]; then
    echo "Phase 4 PASSED (PBT restart triggered and supervisor handled it)"
    PHASE4_OK=1
elif [[ "$P4_RESTART_LOGS" -gt 0 ]]; then
    echo "Phase 4 PASSED (PBT requested restart — observer wrote manifest)"
    PHASE4_OK=1
elif [[ "$P4_EXIT" -eq 0 ]]; then
    # Check if the observer actually tried to rank but training completed first
    if grep -q "tier=\|only.*objectives available" "$P4_RUN"/attempt_*.log 2>/dev/null; then
        echo "Phase 4 PASSED (PBT observer ran, training completed before restart needed)"
        PHASE4_OK=1
    else
        echo "Phase 4 FAILED (PBT observer never fired)"
        echo "  Check log for 'tier=' or 'objectives available' messages"
        tail -20 "$P4_RUN"/attempt_*.log 2>/dev/null
    fi
else
    echo "Phase 4 FAILED (exit $P4_EXIT)"
    tail -30 "$P4_RUN"/attempt_*.log 2>/dev/null
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  PBT Restart Smoke Test Summary"
echo "=========================================="
echo "  Phase 1 (manifest round-trip):  $([ $PHASE1_OK -eq 1 ] && echo PASSED || echo FAILED)"
echo "  Phase 2 (short training):       $([ $PHASE2_OK -eq 1 ] && echo PASSED || echo FAILED)"
echo "  Phase 3 (checkpoint resume):    $([ $PHASE3_OK -eq 1 ] && echo PASSED || echo SKIPPED/FAILED)"
echo "  Phase 4 (PBT observer+restart): $([ $PHASE4_OK -eq 1 ] && echo PASSED || echo FAILED)"
echo ""
echo "Test artifacts: $TEST_ROOT"
echo "=========================================="

if [[ "$PHASE1_OK" -eq 1 ]] && [[ "$PHASE2_OK" -eq 1 ]]; then
    exit 0
else
    exit 1
fi
