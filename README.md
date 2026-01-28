# Labyrinth RL - 3D Ball Maze with Reinforcement Learning

Dự án game Labyrinth 3D chạy trong Apache Zeppelin, sử dụng PySpark làm backend và AngularJS để render 3D. Agent được train bằng Deep Reinforcement Learning theo phong cách sách "Deep Reinforcement Learning Hands-On" của Maxim Lapan.

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                     Apache Zeppelin                          │
├─────────────────────┬───────────────────────────────────────┤
│   %pyspark          │              %angular                  │
│   ┌─────────────┐   │   ┌─────────────────────────────────┐ │
│   │LabyrinthEnv │   │   │  Three.js + Cannon-es          │ │
│   │  - World    │◄──┼───│  - 3D Render                   │ │
│   │  - Physics  │   │   │  - Physics Simulation          │ │
│   │  - Rewards  │   │   │  - WASD/Mouse Controls         │ │
│   └─────────────┘   │   └─────────────────────────────────┘ │
│         │           │                 ▲                      │
│         ▼           │                 │                      │
│   ┌─────────────┐   │      z.angularBind()                  │
│   │  RL Agent   │   │      z.runParagraph()                 │
│   │  - SAC/TD3  │   │                                       │
│   │  - Buffer   │   │                                       │
│   └─────────────┘   │                                       │
└─────────────────────┴───────────────────────────────────────┘
```

## Cấu trúc thư mục

```
Labyrinth_RL/
├── src/
│   ├── labyrinth_env/      # Core game environment
│   │   ├── core/           # Base classes (Entity, World, EventBus)
│   │   ├── entities/       # Game entities (Ball, Wall, Hole, etc.)
│   │   ├── env.py          # Gym-style LabyrinthEnv
│   │   ├── level_spec.py   # Level loader & factory
│   │   └── bridge.py       # Zeppelin binding bridge
│   └── rl/                 # Reinforcement Learning
│       ├── agents/         # Policy implementations (SAC, TD3)
│       ├── buffer.py       # Replay buffer
│       └── train.py        # Training loop
├── notebooks/              # Zeppelin notebooks (JSON)
├── assets/                 # JS libraries, CSS
├── data/
│   ├── levels/             # Level definitions (JSON)
│   ├── models/             # Trained model checkpoints
│   └── logs/               # Training logs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Design Patterns

- **Entity-Component (ECS-lite)**: Quản lý game objects linh hoạt
- **Strategy**: Đổi policy/reward/action mapping dễ dàng
- **State Machine**: Game flow và entity states
- **Observer/Pub-Sub**: Binding giữa backend và frontend
- **Command**: Input handling và replay
- **Factory + Registry**: Tạo entities từ level spec

## Chạy dự án

```bash
# Build và start Zeppelin
docker-compose up --build

# Truy cập Zeppelin UI
open http://localhost:8080
```

## Game Features

- **Physics**: Nghiêng bàn (pitch/roll) để bi lăn theo trọng lực
- **Items**: Coin (+điểm), Key/Lock (mở đường), Teleport, Hole (trap)
- **Levels**: 10 levels từ dễ đến khó với curriculum learning
- **Controls**: WASD/Arrow keys hoặc mouse drag

## RL Training

- **Algorithm**: SAC (Soft Actor-Critic) cho continuous action space
- **Observation**: Ball position/velocity, tilt angles, distances to items
- **Action**: Target tilt (pitch, roll) trong range [-max_tilt, +max_tilt]
- **Reward**: +goal, +coin, +key, -hole, -time_penalty

## Requirements

- Docker & Docker Compose
- Python 3.8+
- PyTorch 2.0+
- NumPy, Gymnasium
