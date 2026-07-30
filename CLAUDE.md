# CLAUDE.md, Project Guide for Claude Code

## Hardware safety

These repos command real machines. Treat every rule here as a hard gate.

- **Never run a command that can move real hardware** unless the user asked for real-hardware
  operation in this session. That includes any service, launch file, or script whose name
  contains `hardware`, `real`, `driver`, or `calibrate`, and anything that opens a `/dev/tty*`
  port. When in doubt, run the simulation variant and say which one you ran.
- **Simulation is the default.** Develop, test, and reproduce in Gazebo first. Confirm which
  environment you are pointed at before publishing to `/cmd_vel` or any controller topic.
- **Always stop what you start.** End every commanded motion sequence with an explicit stop
  (zero twist on `/cmd_vel`, or halt the controller). If a sequence is interrupted or errors
  out, publish the stop yourself before doing anything else.
- **Limits and gains are read-only.** Do not change joint limits, velocity or acceleration
  caps, controller gains, or safety thresholds (URDF, ros2_control YAML, MoveIt
  `joint_limits`, `REAL_SPEED_*` / `REAL_THRESHOLD_*`) without the user signing off on the
  specific values.
- **Shared GPU etiquette.** Training runs take hours on a shared card. Check `nvidia-smi`
  before starting, never kill a process you did not start, and launch training detached
  rather than blocking the session. Prefer the tailnet GPU hosts over the laptop card;
  gpu-node-2 is offline, use gpu-node-1 or gpu-node-3.
- **Do not tear down running work.** Never `docker compose down`, prune, or restart
  containers without checking what is live first (`docker compose ps`, `nvidia-smi`).

## Project Overview

TurtleBot3 Behavior Demos, a ROS 2 (Jazzy) robotics project demonstrating autonomous navigation using behavior trees. A simulated TurtleBot navigates a house environment searching for colored blocks using vision and Nav2-based navigation.

## Repository Layout

- `tb_autonomy/`: ROS 2 package: autonomy behaviors (C++ via BehaviorTree.CPP, Python via py_trees)
- `tb_worlds/`: ROS 2 package: Gazebo simulation worlds, maps, Nav2 config
- `docker/`: Dockerfiles (CPU + GPU) and entrypoint scripts
- `tb_autonomy/bt_xml/`: Behavior tree XML definitions (naive and queue variants)
- `.github/workflows/`: CI: Docker build (`docker.yml`) and pre-commit formatting (`format.yml`)

## Languages

- **C++**: Core autonomy node and behavior tree plugins (`tb_autonomy/src/`, `tb_autonomy/include/`)
- **Python**: Autonomy node, behavior library, launch files (`tb_autonomy/python/`, `tb_autonomy/scripts/`, `*/launch/`)
- **CMake**: Build system (`CMakeLists.txt` files, ament_cmake)
- **XML/SDF/Xacro**: Behavior trees and simulation models

## Build System

ROS 2 colcon workspace. Build with:

```bash
colcon build --symlink-install
source install/setup.bash
```

## Linting & Formatting

Pre-commit hooks are configured (`.pre-commit-config.yaml`):

```bash
pre-commit run -a          # Run all hooks
pre-commit install         # Install as git hook
```

Hooks include:
- `black`: Python formatter
- `clang-format` v18: C/C++ formatter
- `codespell`: Spell checker (ignore list: `atleast,inout,ether`)
- `yamllint`: YAML linter
- `markdown-link-check`: Markdown link validator
- Standard pre-commit checks (AST, YAML, merge conflicts, etc.)

## Docker

```bash
docker compose up demo-world         # Launch simulation
docker compose up demo-behavior-py   # Python behavior demo
docker compose up demo-behavior-cpp  # C++ behavior demo
```

## Documentation Standards

All architecture and data-flow diagrams **must use Mermaid**, no ASCII art or image files.
This applies to README.md and all other markdown in the repo.

- Follow the `mermaid-diagrams` skill for dark-mode-safe styling (classDef on every node,
  `rgba()` fills for `rect`, plain-text subgraph labels). Do not restate its palette here.

## Git Workflow

The `main` branch is protected, direct pushes are blocked for everyone including admins.
All changes must go through a pull request.

```bash
# 1. Create a feature branch
git checkout -b feat/my-change

# 2. Commit work normally
git add <files>
git commit -m "feat: description"

# 3. Push and open a PR
git push -u origin feat/my-change
gh pr create --title "Short title" --body "$(cat <<'EOF'
## Summary
- bullet points

## Test plan
- [ ] tests pass

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"

# 4. Enable auto-merge (merges immediately, no reviewer required)
gh pr merge --auto --squash
```

Auto-merge is enabled at the repo level: once any required CI checks pass the PR merges automatically.
Use `--squash` (default), `--merge`, or `--rebase` depending on the change.

## Issue Tracking

Issues live in Jira, project AURA at https://aegean-ai.atlassian.net. Use the Atlassian MCP tools (`mcp__plugin_atlassian_atlassian__*`): `searchJiraIssuesUsingJql` to find work, `getJiraIssue` to read, `createJiraIssue` to file, `addCommentToJiraIssue` to comment, `transitionJiraIssue` to change status. Search for an existing issue before filing a new one.

beads, the `bd` CLI, and the per-repo Dolt server are retired. Never run `bd`, never start a Dolt server, and never recreate a `.beads/` directory. The leftover `.beads-archive.zip` and `.dolt/` in this repo are historical artifacts, not live infrastructure.

**Before closing any issue**, add a comment that records:
- What was done (key changes made, files modified)
- Root causes found (for bugs)
- Lessons learned (gotchas, non-obvious behaviour, useful debugging insights)
- Verification performed (tests run, output observed)

This makes closed issues a searchable knowledge base for future debugging sessions.

## Robot Commands via ros-mcp-server

Always use the ros-mcp-server MCP tools (`publish_for_durations`, `publish_once`, etc.) to command the robot, not `docker exec` with `ros2 topic pub`.

Rosbridge has a DDS publisher discovery delay (~5 seconds). When publishing to `/cmd_vel`, prepend 5 warmup messages (1 second each, zero velocity) before the actual motion commands. Without this, `ros_gz_bridge` won't discover the publisher in time and the commands are lost.

```
# Example: rotate the robot
publish_for_durations(
  topic="/cmd_vel",
  msg_type="geometry_msgs/msg/TwistStamped",
  messages=[
    # 5× warmup (zero velocity, gives DDS time to discover publisher)
    {"header": {"frame_id": "base_link"}, "twist": {"angular": {"z": 0}}},
    {"header": {"frame_id": "base_link"}, "twist": {"angular": {"z": 0}}},
    {"header": {"frame_id": "base_link"}, "twist": {"angular": {"z": 0}}},
    {"header": {"frame_id": "base_link"}, "twist": {"angular": {"z": 0}}},
    {"header": {"frame_id": "base_link"}, "twist": {"angular": {"z": 0}}},
    # Actual motion
    {"header": {"frame_id": "base_link"}, "twist": {"angular": {"z": 0.5}}},
    ...
    # Stop
    {"header": {"frame_id": "base_link"}, "twist": {"angular": {"z": 0}}},
  ],
  durations=[1, 1, 1, 1, 1, 1, ..., 0.5]
)
```

## Key Configuration

Runtime configuration lives in `.env` at the repo root: ROS distro, TurtleBot model, behavior
tree type, vision toggle, detector type, and target colour/object. Read that file before
launching rather than trusting a copy here, and never hardcode a value it already defines.
Duplicating them in this file is how they drift.
