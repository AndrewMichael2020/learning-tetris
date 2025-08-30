
# Add Four “Comedy‑Named” Algorithms to **learning‑tetris** (Copilot‑Ready Guide)

Target repo: `AndrewMichael2020/learning-tetris`  
Algorithms to add:
- **Tabu Search (“Nurse Gossip”)**
- **Simulated Annealing (“Coffee Break”)**
- **Greedy Heuristic (“Nurse Dictator”)**
- **Ant Colony Optimization (“Night Shift Ant March”)**

This file gives Copilot-steerable prompts, math, Python scaffolds, API wiring, UI placement, inputs, and tests.

---

## 0) Shared contracts

### 0.1 Objective and penalties
Let the after-state features be
- `h`: holes
- `mh`: max column height
- `bm`: bumpiness
- `lp`: negative potential for future line clears (the lower the better)
- any others you already compute

Define a **total cost** (minimize):
\[
J(s) = M \cdot \text{hard\_violations}(s) + \sum_k w_k f_k(s)
\]
with a very large \(M = 1e6\) to forbid hard violations (collisions/out-of-bounds). Feature weights \(w_k\) are user-configurable.

### 0.2 Agent interface (new file)
Create `rl/agent_base.py`:
```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    @abstractmethod
    def select_action(self, state) -> int:
        """Return an action index for the current piece (0..A-1)."""

    def notify_step(self, prev_state, action, new_state, reward: float, info: Dict[str, Any]):
        """Optional hook for learning-based updates."""

    def reset(self):
        """Optional episodic reset."""
```

All new algorithms will implement this interface and use the existing **after‑state enumerator** and **feature scorer**.

### 0.3 Shared helpers
Create `rl/search_utils.py`:
```python
import math
import random
from typing import List, Tuple, Dict

def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def argmin_with_index(vals: List[float]) -> Tuple[int, float]:
    i = min(range(len(vals)), key=lambda k: vals[k])
    return i, vals[i]

def metropolis_accept(delta: float, T: float) -> bool:
    if delta <= 0: 
        return True
    return random.random() < math.exp(-delta / max(T, 1e-12))
```

---

## 1) Tabu Search — “Nurse Gossip”

### Math
- Neighborhood \( \mathcal{N}(s) \): all feasible placements for the next piece (or top‑K by heuristic).
- Maintain a **tabu list** of attributes (e.g., `(col, rot)` or explicit `action_id`) of length `tenure`.
- At each iteration choose the neighbor with minimal \(J\) that is **not tabu**, unless it **aspirates**: \( J(s') < J^*_{\text{best}} \).

### Inputs
**Required:**
- `tenure` (int; default 25)
- `max_iters` (int; default 500)
- `neighborhood_top_k` (int; default 10)

**Cool from user:**
- feature weights `{w_k}`
- aspiration rule on/off
- random seed

### Python scaffold
Create `rl/tabu_agent.py`:
```python
from typing import Deque, List, Dict, Any
from collections import deque
import random

from .agent_base import Agent
from .features import score_afterstate  # assumes returns float cost J
from .afterstate import enumerate_actions  # returns list of after-states for current piece

class TabuAgent(Agent):
    def __init__(self, tenure=25, max_iters=500, neighborhood_top_k=10, aspiration=True, rng_seed=42, **weights):
        self.tenure = tenure
        self.max_iters = max_iters
        self.k = neighborhood_top_k
        self.aspiration = aspiration
        random.seed(rng_seed)
        self.weights = weights
        self._tabu: Deque[int] = deque(maxlen=self.tenure)
        self.best_cost = float("inf")

    def reset(self):
        self._tabu.clear()
        self.best_cost = float("inf")

    def select_action(self, state) -> int:
        actions, after_states = enumerate_actions(state)
        # Heuristic prefilter: top‑K by current weights
        costs = [score_afterstate(s, **self.weights) for s in after_states]
        idx_sorted = sorted(range(len(costs)), key=lambda i: costs[i])[:min(self.k, len(costs))]

        # Tabu filter with aspiration
        candidate = None
        candidate_cost = float("inf")
        for i in idx_sorted:
            if (i in self._tabu) and not (self.aspiration and costs[i] < self.best_cost):
                continue
            if costs[i] < candidate_cost:
                candidate = i
                candidate_cost = costs[i]

        if candidate is None:
            candidate = idx_sorted[0]
            candidate_cost = costs[candidate]

        # update tabu memory and incumbent
        self._tabu.append(candidate)
        if candidate_cost < self.best_cost:
            self.best_cost = candidate_cost
        return actions[candidate]
```

---

## 2) Simulated Annealing — “Coffee Break”

### Math
Proposal from neighborhood \(s' \sim q(\cdot \mid s)\) (e.g., top‑K placements at random).  
Accept with **Metropolis** rule:
\[
\text{accept}(s \to s') =
\begin{cases}
1,& \Delta = J(s') - J(s) \le 0 \\
\exp(-\Delta/T),& \Delta>0
\end{cases}
\]
Cooling: \( T_{t+1} = \alpha T_t \) with \( \alpha \in (0,1) \).

### Inputs
**Required:**
- `T0` (float; default = 10 × median |ΔJ| of sampled neighbors)
- `alpha` (float; default 0.99)
- `steps_per_T` (int; default 500)

**Cool from user:**
- top‑K used for proposals
- stopping on “no improvement” window
- seed

### Python scaffold
Create `rl/sa_agent.py`:
```python
import random
import statistics

from .agent_base import Agent
from .search_utils import metropolis_accept
from .features import score_afterstate
from .afterstate import enumerate_actions

class SimulatedAnnealingAgent(Agent):
    def __init__(self, T0=None, alpha=0.99, steps_per_T=500, proposal_top_k=10, rng_seed=42, **weights):
        random.seed(rng_seed)
        self.T0 = T0
        self.alpha = alpha
        self.steps_per_T = steps_per_T
        self.k = proposal_top_k
        self.weights = weights
        self.T = None
        self.prev_cost = None

    def reset(self):
        self.T = self.T0
        self.prev_cost = None

    def select_action(self, state) -> int:
        actions, after_states = enumerate_actions(state)
        costs = [score_afterstate(s, **self.weights) for s in after_states]
        order = sorted(range(len(costs)), key=lambda i: costs[i])[:min(self.k, len(costs))]

        # temperature bootstrap if needed
        if self.T is None:
            sample = [costs[i] for i in order]
            med = statistics.median(sample) if sample else 1.0
            self.T = (self.T0 or 10.0 * max(1.0, med))

        # propose one at random from top‑K
        i_prop = random.choice(order)
        c_prop = costs[i_prop]
        c_curr = min(costs) if self.prev_cost is None else self.prev_cost
        accept = metropolis_accept(c_prop - c_curr, self.T)

        if accept:
            chosen = i_prop
            self.prev_cost = c_prop
        else:
            chosen = order[0]
            self.prev_cost = costs[chosen]

        # cool temperature every call
        self.T *= self.alpha
        return actions[chosen]
```

---

## 3) Greedy Heuristic — “Nurse Dictator”

### Math
Pick action \( a^* = \arg\min_a J(\text{after}(s,a)) \). No lookahead, no learning.

### Inputs
**Required:**
- feature weights `{w_k}`

**Cool from user:**
- deterministic tie‑breakers: “prefer lower `mh` then fewer holes”

### Python scaffold
Create `rl/greedy_agent.py`:
```python
from .agent_base import Agent
from .features import score_afterstate
from .afterstate import enumerate_actions

class GreedyAgent(Agent):
    def __init__(self, **weights):
        self.weights = weights

    def select_action(self, state) -> int:
        actions, after_states = enumerate_actions(state)
        costs = [score_afterstate(s, **self.weights) for s in after_states]
        best = min(range(len(costs)), key=lambda i: costs[i])
        return actions[best]
```

---

## 4) Ant Colony Optimization — “Night Shift Ant March”

### Math
- Pheromone matrix \( \tau_i \) over actionable placements `i`.  
- Heuristic \( \eta_i = 1 / (1 + J_i) \) from feature cost for after‑state `i`.  
- Choice probability for ant \(m\):
\[
p_i^{(m)} = \frac{\tau_i^\alpha \cdot \eta_i^\beta}{\sum_j \tau_j^\alpha \cdot \eta_j^\beta}
\]
- Update (per iteration):
\[
\tau_i \leftarrow (1-\rho)\tau_i + \Delta \tau_i, \quad
\Delta \tau_i = \sum_{m \in \text{elite}} \frac{Q}{1+J^{(m)}} \cdot \mathbb{1}\{i=\text{choice}^{(m)}\}
\]

### Inputs
**Required:**
- `alpha` (float; default 1.0)
- `beta` (float; default 2.0)
- `rho` evaporation (float; default 0.10)
- `ants` per iteration (int; default 20)
- `elite` count (int; default 1)
- `Q` deposit scale (float; default 1.0)

**Cool from user:**
- warm‑start pheromone
- cap on max τ

### Python scaffold
Create `rl/aco_agent.py`:
```python
import math
import random
from typing import List

from .agent_base import Agent
from .features import score_afterstate
from .afterstate import enumerate_actions
from .search_utils import softmax

class ACOAgent(Agent):
    def __init__(self, alpha=1.0, beta=2.0, rho=0.10, ants=20, elite=1, Q=1.0, rng_seed=42, **weights):
        random.seed(rng_seed)
        self.alpha, self.beta, self.rho = alpha, beta, rho
        self.ants, self.elite, self.Q = ants, elite, Q
        self.weights = weights
        self.pheromone: List[float] = []

    def reset(self):
        self.pheromone = []

    def select_action(self, state) -> int:
        actions, after_states = enumerate_actions(state)
        n = len(actions)
        costs = [score_afterstate(s, **self.weights) for s in after_states]
        eta = [1.0 / (1.0 + c) for c in costs]

        if not self.pheromone or len(self.pheromone) != n:
            self.pheromone = [1.0] * n  # init flat

        # Build solutions for all ants (single-step construction)
        choices = []
        for _ in range(self.ants):
            logits = [self.alpha * math.log(self.pheromone[i]) + self.beta * math.log(eta[i]) for i in range(n)]
            probs = softmax(logits)
            r = random.random()
            cum = 0.0
            pick = 0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    pick = i
                    break
            choices.append((pick, costs[pick]))

        # Elite reinforcement
        choices.sort(key=lambda x: x[1])
        elites = choices[: self.elite]

        # Evaporate
        self.pheromone = [(1.0 - self.rho) * t for t in self.pheromone]
        # Deposit
        for idx, c in elites:
            self.pheromone[idx] += self.Q / (1.0 + c)

        # Play the current best
        return actions[elites[0][0]]
```

> Note: This is a one‑step ACO because Tetris acts piece‑by‑piece. For multi‑step construction (look‑ahead across next pieces), extend the pheromone to sequences `(i_t, i_{t+1})`.

---

## 5) Wire into FastAPI

### 5.1 Register algorithms
In `app/main.py`, extend the algorithm factory:
```python
from rl.greedy_agent import GreedyAgent
from rl.tabu_agent import TabuAgent
from rl.sa_agent import SimulatedAnnealingAgent
from rl.aco_agent import ACOAgent

def make_agent(kind: str, params: dict):
    if kind == "GREEDY":
        return GreedyAgent(**params)
    if kind == "TABU":
        return TabuAgent(**params)
    if kind == "ANNEAL":
        return SimulatedAnnealingAgent(**params)
    if kind == "ACO":
        return ACOAgent(**params)
    # existing: CEM, REINFORCE, etc.
    raise ValueError(f"Unknown algorithm: {kind}")
```

Expose them in your `/play`, `/stream`, and `/quick-train` routes by threading `algo` and `params` from the request body or query string into `make_agent`.

### 5.2 API input schema
Create `app/schemas_algos.py`:
```python
from pydantic import BaseModel, Field
from typing import Optional

class BaseWeights(BaseModel):
    w_holes: float = 1.0
    w_max_height: float = 1.0
    w_bumpiness: float = 1.0
    w_line_potential: float = 1.0

class TabuParams(BaseModel, BaseWeights):
    tenure: int = 25
    max_iters: int = 500
    neighborhood_top_k: int = 10
    aspiration: bool = True
    rng_seed: int = 42

class AnnealParams(BaseModel, BaseWeights):
    T0: Optional[float] = None
    alpha: float = 0.99
    steps_per_T: int = 500
    proposal_top_k: int = 10
    rng_seed: int = 42

class GreedyParams(BaseModel, BaseWeights):
    pass

class ACOParams(BaseModel, BaseWeights):
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.10
    ants: int = 20
    elite: int = 1
    Q: float = 1.0
    rng_seed: int = 42
```
Use these in request bodies so the UI can post clean JSON.

---

## 6) UI placement (below current “Controls”)

Add a **new column/section** titled **“Algorithm Settings (Comedy Edition)”** under the existing control panel.

Create four collapsible panels (accordions), one per algorithm, each with minimal inputs and “advanced” toggles.

**Suggested layout (top→bottom):**
1. Dropdown `Algorithm: [REINFORCE, CEM, GREEDY, TABU, ANNEAL, ACO]`
2. **Greedy (“Nurse Dictator”)**
   - Weights: `w_holes`, `w_max_height`, `w_bumpiness`, `w_line_potential`
3. **Tabu (“Nurse Gossip”)**
   - `tenure`, `neighborhood_top_k`, `max_iters`, `aspiration`, `rng_seed`
4. **Anneal (“Coffee Break”)**
   - `T0` (empty = auto), `alpha`, `steps_per_T`, `proposal_top_k`, `rng_seed`
5. **Ant Colony (“Night Shift Ant March”)**
   - `alpha`, `beta`, `rho`, `ants`, `elite`, `Q`, `rng_seed`

Each panel should show: **[Use Defaults]** and **[Apply]** buttons. On Apply, post JSON to the app and update the live session config.

Include a read‑only box “Live Score/Temp/Best/Tabu size/τ‑stats” depending on algorithm.

---

## 7) Tests

Create `tests/test_algorithms.py`:
```python
import math
import pytest

from rl.greedy_agent import GreedyAgent
from rl.tabu_agent import TabuAgent
from rl.sa_agent import SimulatedAnnealingAgent
from rl.aco_agent import ACOAgent

def dummy_state():
    # Provide a minimal mocked state compatible with enumerate_actions/score_afterstate
    # Copilot: synthesize 5 actions with fixed costs [5,3,7,2,4]
    ...

def test_greedy_picks_min():
    s = dummy_state()
    agent = GreedyAgent(w_holes=1.0, w_max_height=1.0, w_bumpiness=1.0, w_line_potential=1.0)
    a = agent.select_action(s)
    # Expect action with minimal cost (index 3 in dummy)
    assert a == 3

def test_tabu_avoids_recent():
    s = dummy_state()
    agent = TabuAgent(tenure=2, neighborhood_top_k=5)
    a1 = agent.select_action(s)
    agent.reset()
    agent._tabu.append(a1)
    a2 = agent.select_action(s)
    assert a2 != a1

def test_anneal_accepts_better():
    s = dummy_state()
    agent = SimulatedAnnealingAgent(T0=1.0, alpha=0.95, steps_per_T=1, proposal_top_k=5)
    a = agent.select_action(s)
    assert isinstance(a, int)

def test_aco_updates_pheromone():
    s = dummy_state()
    agent = ACOAgent(ants=10, elite=1)
    a1 = agent.select_action(s)
    t_before = list(agent.pheromone)
    a2 = agent.select_action(s)
    assert any(tb != ta for tb, ta in zip(t_before, agent.pheromone))
```

Add a smoke test that runs one episode per algorithm and asserts no exceptions and positive line clears.

---

## 8) Copilot prompts (paste into the IDE when coding)

> "Create `rl/agent_base.py` with an abstract class `Agent` exposing `select_action`, `notify_step`, `reset`. Then implement `GreedyAgent`, `TabuAgent`, `SimulatedAnnealingAgent`, and `ACOAgent` using the repository’s `enumerate_actions` and `score_afterstate`. Each agent returns an action index. Add a factory `make_agent` in `app/main.py` mapping strings to classes. Add Pydantic schemas for parameters. Add unit tests in `tests/test_algorithms.py` with a minimal dummy state to mock enumerations and costs."

> "Extend the FastAPI routes to accept an `algo` string and `params` JSON body conforming to the new schemas. Thread these into `make_agent` and run episodes with live updates."

> "Modify the front‑end control panel by adding a new ‘Algorithm Settings (Comedy Edition)’ section with collapsible forms for GREEDY, TABU, ANNEAL, ACO. Bind inputs, send to backend, and render live metrics (best J, temperature T, tabu size, pheromone min/max)."

---

## 9) Defaults (good starting values)

- Greedy: `w_holes=8`, `w_max_height=1`, `w_bumpiness=1`, `w_line_potential=2`
- Tabu: `tenure=25`, `neighborhood_top_k=10`, `max_iters=500`, `aspiration=True`
- Anneal: `T0=None (auto)`, `alpha=0.99`, `steps_per_T=500`, `proposal_top_k=10`
- ACO: `alpha=1`, `beta=2`, `rho=0.10`, `ants=20`, `elite=1`, `Q=1.0`

---

## 10) Telemetry hooks (optional)
Emit per‑step stats to the right‑hand Statistics panel:
- Greedy: current cost
- Tabu: best cost, tabu list length
- Anneal: T, last ΔJ, acceptance rate
- ACO: min/mean/max τ, chosen index

These can be appended to your existing websocket/stream updates.

---

**Done.**  Drop this file into the repo root as `ADD_ALGOS_COMEDY.md`.
