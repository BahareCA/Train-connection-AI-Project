# Train Journey Planner, Multi-Criteria Route Search over a Railway Timetable

A Python program that finds the best train connection between two stations from a real railway timetable, using **Dijkstra's algorithm**. It can optimise for four different goals, shortest distance, fewest stops, cheapest ticket, or earliest arrival, and handles practical details like buying a new ticket when you change trains, waiting time between connections, and journeys that run past midnight.

## Overview

The timetable is loaded from CSV and turned into a **graph**: each station is a node, and each leg between two consecutive stops on the same train is an edge carrying its train number, departure and arrival times, and distance. Given a start and destination station, the planner searches this graph for the optimal route under a chosen cost function, then writes out the connection (which trains to take and where to change) together with its cost.

## What it demonstrates

- Building a graph model from tabular timetable data (pandas)
- Dijkstra's shortest-path search with a priority queue (`heapq`)
- A clean abstraction over **four interchangeable cost functions** selected per query
- Real-world modelling: ticket changes on transfers, station waiting times, and overnight day-rollover in the time arithmetic
- Batch processing — reads many problems from CSV and writes structured results back to CSV

## The four optimisation modes

| Mode | What it minimises | How the cost is defined |
|---|---|---|
| `distance` | Total kilometres travelled | Sum of each leg's distance |
| `stops` | Number of stops/hops | One unit per leg |
| `price` | Number of tickets bought | A new ticket only when you switch to a different train; staying on the same train is free |
| `arrivaltime HH:MM:SS` | Time of arrival from a given start time | Waiting time at each station plus travel time, with overnight journeys rolling into the next day |

## How it works

1. **Build the graph** : group the schedule by train, sort each train's stops in order, and create one edge per consecutive pair of stops.
2. **Search** : run Dijkstra from the start station, expanding the lowest-cost route first according to the selected cost function, until the destination is reached.
3. **Reconstruct and report** : turn the resulting path into a readable connection string (train numbers and stop sequence numbers, with change-overs marked) and compute the total cost; for arrival-time queries it also returns the final clock time and day offset.

## Repository contents

| File | What it is |
|---|---|
| `FinalSolution.py` | The planner — graph building, Dijkstra, cost functions, and CSV I/O |
| `problems.csv` | Query set: problem number, from/to stations, cost function, and which schedule to use |
| `solutions.csv` | Generated output: the chosen connection and its cost per problem |
| `Task.docx` | Original assignment brief |

## Input and output format

- **`problems.csv`** columns: `ProblemNo`, `FromStation`, `ToStation`, `CostFunction`, `Schedule`. The `CostFunction` is one of `distance`, `stops`, `price`, or `arrivaltime HH:MM:SS`.
- **Schedule CSV** columns: `Train No.`, `islno` (stop sequence), `station Code`, `Departure time`, `Arrival time`, `Distance`.
- **`solutions.csv`** columns: `ProblemNo`, `Connection`, `Cost`.

## Run it

```bash
# Python 3
pip install pandas
python FinalSolution.py   # reads problems.csv (+ the schedule files), writes solutions.csv
```

> **Note:** `main()` loads the timetables `schedule.csv` and `mini-schedule.csv`, which `problems.csv` refers to in its `Schedule` column. Add those two CSVs to the repo so the project runs out of the box for anyone who clones it.

## Possible extensions

Compare results across the four cost functions for the same trip, add a small visualisation of the chosen route, or expose the planner through a simple command-line interface or web form.

---

*Suggested GitHub "About" description:* **Python train journey planner using Dijkstra's algorithm over a railway timetable, optimising for distance, stops, price, or arrival time.**
*Suggested topics:* `python` · `artificial-intelligence` · `dijkstra` · `shortest-path` · `graph-search` · `pandas` · `route-planning`
