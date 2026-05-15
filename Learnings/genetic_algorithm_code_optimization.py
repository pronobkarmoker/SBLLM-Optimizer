"""
============================================================
  GENETIC ALGORITHM — CODE / PARAMETER OPTIMIZATION
============================================================
  Problem: Optimize hyperparameters of a Machine Learning
           pipeline to maximize model accuracy.

  Dataset: Auto-generated classification dataset
           (1000 samples, 10 features, 2 classes)

  What GA Optimizes:
    - learning_rate   : float  [0.0001 – 0.5]
    - n_estimators    : int    [10 – 300]
    - max_depth       : int    [1 – 20]
    - min_samples_split: int   [2 – 20]
    - subsample       : float  [0.5 – 1.0]

  GA Steps:
    1. Encode hyperparameters as chromosomes
    2. Initialize random population
    3. Evaluate fitness (cross-val accuracy)
    4. Tournament selection
    5. Uniform crossover
    6. Gaussian / random mutation
    7. Elitism + replacement
    8. Repeat for N generations
============================================================
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────
#  SEED for reproducibility
# ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────
#  DATASET — Auto-generated Classification Dataset
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("  GENERATING DATASET")
print("=" * 60)

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=SEED
)

print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}")
print(f"  Classes  : {len(set(y))}  (Binary Classification)")
print(f"  Class 0  : {sum(y == 0)}  samples")
print(f"  Class 1  : {sum(y == 1)}  samples")
print()


# ─────────────────────────────────────────────────────────
#  HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────────────────
PARAM_SPACE = {
    "learning_rate":      (0.0001, 0.5),     # float
    "n_estimators":       (10, 300),          # int
    "max_depth":          (1, 20),            # int
    "min_samples_split":  (2, 20),            # int
    "subsample":          (0.5, 1.0),         # float
}

PARAM_NAMES = list(PARAM_SPACE.keys())


# ─────────────────────────────────────────────────────────
#  CHROMOSOME ENCODING / DECODING
#
#  A chromosome = [learning_rate, n_estimators,
#                  max_depth, min_samples_split, subsample]
# ─────────────────────────────────────────────────────────
def random_chromosome():
    """Create one random set of hyperparameters."""
    return [
        round(random.uniform(*PARAM_SPACE["learning_rate"]), 5),
        random.randint(*PARAM_SPACE["n_estimators"]),
        random.randint(*PARAM_SPACE["max_depth"]),
        random.randint(*PARAM_SPACE["min_samples_split"]),
        round(random.uniform(*PARAM_SPACE["subsample"]), 4),
    ]


def decode(chromosome):
    """Map chromosome list → named parameter dict."""
    return {
        "learning_rate":     chromosome[0],
        "n_estimators":      int(chromosome[1]),
        "max_depth":         int(chromosome[2]),
        "min_samples_split": int(chromosome[3]),
        "subsample":         chromosome[4],
    }


def clamp(chromosome):
    """Ensure all genes stay within valid bounds."""
    bounds = list(PARAM_SPACE.values())
    clamped = []
    for i, (val, (lo, hi)) in enumerate(zip(chromosome, bounds)):
        val = max(lo, min(hi, val))
        # int genes
        if i in [1, 2, 3]:
            val = int(round(val))
        else:
            val = round(float(val), 5)
        clamped.append(val)
    return clamped


# ─────────────────────────────────────────────────────────
#  FITNESS FUNCTION
#  Fitness = 5-fold cross-validation accuracy
# ─────────────────────────────────────────────────────────
fitness_cache = {}

def fitness(chromosome):
    key = tuple(chromosome)
    if key in fitness_cache:
        return fitness_cache[key]

    params = decode(chromosome)
    model = GradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        subsample=params["subsample"],
        random_state=SEED
    )
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    acc = round(scores.mean(), 6)
    fitness_cache[key] = acc
    return acc


# ─────────────────────────────────────────────────────────
#  STEP 1: INITIALIZE POPULATION
# ─────────────────────────────────────────────────────────
def init_population(size):
    return [random_chromosome() for _ in range(size)]


# ─────────────────────────────────────────────────────────
#  STEP 2: TOURNAMENT SELECTION
# ─────────────────────────────────────────────────────────
def tournament_selection(population, fitnesses, k=5):
    """Select k random individuals, return the best."""
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx][:]


# ─────────────────────────────────────────────────────────
#  STEP 3: UNIFORM CROSSOVER
# ─────────────────────────────────────────────────────────
def uniform_crossover(parent1, parent2):
    """Each gene independently taken from parent1 or parent2."""
    child1, child2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1); child2.append(g2)
        else:
            child1.append(g2); child2.append(g1)
    return child1, child2


# ─────────────────────────────────────────────────────────
#  STEP 4: GAUSSIAN MUTATION
# ─────────────────────────────────────────────────────────
def mutate(chromosome, mutation_rate=0.2):
    """
    Gaussian noise on float genes,
    ±random integer step on int genes.
    """
    mutated = chromosome[:]
    bounds = list(PARAM_SPACE.values())

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            lo, hi = bounds[i]
            if i in [1, 2, 3]:   # integer genes
                step = random.randint(1, max(1, (hi - lo) // 5))
                mutated[i] += random.choice([-step, step])
            else:                 # float genes
                sigma = (hi - lo) * 0.1
                mutated[i] += random.gauss(0, sigma)

    return clamp(mutated)


# ─────────────────────────────────────────────────────────
#  STEP 5: GENETIC ALGORITHM — MAIN LOOP
# ─────────────────────────────────────────────────────────
def genetic_algorithm(
    pop_size=30,
    generations=20,
    mutation_rate=0.2,
    elite_size=4,
    tournament_k=5,
):
    print("=" * 60)
    print("  GENETIC ALGORITHM — HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"  Population Size : {pop_size}")
    print(f"  Generations     : {generations}")
    print(f"  Mutation Rate   : {mutation_rate}")
    print(f"  Elite Size      : {elite_size}")
    print(f"  Tournament K    : {tournament_k}")
    print(f"  Fitness Metric  : 5-Fold CV Accuracy")
    print()

    population = init_population(pop_size)
    best_scores = []
    avg_scores = []
    overall_best = None
    overall_best_score = 0.0

    for gen in range(1, generations + 1):
        # Evaluate fitness for all individuals
        fitnesses = [fitness(c) for c in population]

        gen_best_score = max(fitnesses)
        gen_avg_score = sum(fitnesses) / len(fitnesses)
        best_scores.append(gen_best_score)
        avg_scores.append(gen_avg_score)

        best_idx = fitnesses.index(gen_best_score)
        if gen_best_score > overall_best_score:
            overall_best_score = gen_best_score
            overall_best = population[best_idx][:]

        print(f"  Gen {gen:>3}/{generations} | "
              f"Best Acc: {gen_best_score:.4f} | "
              f"Avg Acc: {gen_avg_score:.4f} | "
              f"Params: lr={population[best_idx][0]:.4f}, "
              f"n_est={int(population[best_idx][1])}, "
              f"depth={int(population[best_idx][2])}")

        # ── Elitism: keep top individuals ──────────────────
        sorted_pop = [x for _, x in sorted(
            zip(fitnesses, population), key=lambda p: p[0], reverse=True
        )]
        new_population = sorted_pop[:elite_size]

        # ── Fill new generation ────────────────────────────
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses, tournament_k)
            p2 = tournament_selection(population, fitnesses, tournament_k)
            c1, c2 = uniform_crossover(p1, p2)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            new_population.extend([c1, c2])

        population = new_population[:pop_size]

    return overall_best, overall_best_score, best_scores, avg_scores


# ─────────────────────────────────────────────────────────
#  RUN GA
# ─────────────────────────────────────────────────────────
best_chrom, best_acc, best_scores, avg_scores = genetic_algorithm(
    pop_size=30,
    generations=20,
    mutation_rate=0.2,
    elite_size=4,
    tournament_k=5,
)


# ─────────────────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────────────────
best_params = decode(best_chrom)

print()
print("=" * 60)
print("  OPTIMIZATION COMPLETE — BEST HYPERPARAMETERS FOUND")
print("=" * 60)
for k, v in best_params.items():
    print(f"  {k:<22} : {v}")
print(f"\n  Best CV Accuracy    : {best_acc * 100:.4f} %")
print()


# ─────────────────────────────────────────────────────────
#  BASELINE COMPARISON
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("  BASELINE vs GA-OPTIMIZED MODEL")
print("=" * 60)

# Default model (sklearn defaults)
default_model = GradientBoostingClassifier(random_state=SEED)
default_scores = cross_val_score(default_model, X, y, cv=5, scoring="accuracy")
default_acc = default_scores.mean()
print(f"  Default Model Accuracy  : {default_acc * 100:.4f} %")

# GA-optimized model
opt_model = GradientBoostingClassifier(
    **best_params, random_state=SEED
)
opt_scores = cross_val_score(opt_model, X, y, cv=5, scoring="accuracy")
opt_acc = opt_scores.mean()
print(f"  GA-Optimized Accuracy   : {opt_acc * 100:.4f} %")
print(f"  Improvement             : +{(opt_acc - default_acc) * 100:.4f} %")
print()


# ─────────────────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Genetic Algorithm — Hyperparameter Optimization", fontsize=15, fontweight="bold")

# ── Plot 1: Fitness over Generations ──────────────────────
gens = range(1, len(best_scores) + 1)
axes[0].plot(gens, [s * 100 for s in best_scores], 'o-',
             color="#2ecc71", linewidth=2, markersize=5, label="Best Accuracy")
axes[0].plot(gens, [s * 100 for s in avg_scores], 's--',
             color="#3498db", linewidth=2, markersize=4, label="Avg Accuracy")
axes[0].set_title("Fitness Evolution Across Generations", fontweight="bold")
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("CV Accuracy (%)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(
    min(min(s * 100 for s in avg_scores) - 1, 85),
    max(max(s * 100 for s in best_scores) + 1, 95)
)

# ── Plot 2: Default vs Optimized ──────────────────────────
labels = ["Default\nModel", "GA-Optimized\nModel"]
accs = [default_acc * 100, opt_acc * 100]
colors = ["#e74c3c", "#2ecc71"]
bars = axes[1].bar(labels, accs, color=colors, width=0.4, edgecolor="black", linewidth=0.8)
for bar, acc in zip(bars, accs):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{acc:.3f}%",
        ha="center", va="bottom", fontweight="bold", fontsize=11
    )
axes[1].set_title("Default vs GA-Optimized Accuracy", fontweight="bold")
axes[1].set_ylabel("CV Accuracy (%)")
axes[1].set_ylim(min(accs) - 2, max(accs) + 2)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("ga_optimization_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Plot saved as: ga_optimization_results.png")
