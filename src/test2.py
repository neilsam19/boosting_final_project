import numpy as np
from data import prepare_data
from adaboost import AdaBoost
from modifications import AdaBoostClipped


def evaluate(dataset, noise, seeds, cap=0.01):
    vanilla_accs = []
    clipped_accs = []

    # 🔥 choose number of rounds based on dataset
    if dataset in ["moons", "classification"]:
        T = 20
    else:
        T = 50  # real datasets need more rounds

    for seed in seeds:
        X_train, X_test, y_train, y_test = prepare_data(
            dataset=dataset,
            noise_level=noise,
            random_state=seed
        )

        # Vanilla
        model1 = AdaBoost(T=T)
        model1.fit(X_train, y_train)
        vanilla_accs.append((model1.predict(X_test) == y_test).mean())

        # Clipped
        model2 = AdaBoostClipped(T=T, cap=cap)
        model2.fit(X_train, y_train)
        clipped_accs.append((model2.predict(X_test) == y_test).mean())

    return (
        np.mean(vanilla_accs), np.std(vanilla_accs),
        np.mean(clipped_accs), np.std(clipped_accs)
    )


# seeds for averaging
seeds = [0, 1, 2, 3, 4]

datasets = ["moons", "classification", "digits", "wine"]

for dataset in datasets:
    print(f"\n==============================")
    print(f"Dataset: {dataset}")
    print(f"==============================")

    for noise in [0.0, 0.05, 0.1, 0.15, 0.2]:
        v_mean, v_std, c_mean, c_std = evaluate(dataset, noise, seeds)

        diff = c_mean - v_mean

        print(
            f"noise={noise:.2f} | "
            f"vanilla={v_mean:.4f} | "
            f"clipped={c_mean:.4f} | "
            f"diff={diff:+.4f}"
        )

# DEBUG: inspect max weights for one run
X_train, X_test, y_train, y_test = prepare_data(
    dataset="digits",
    noise_level=0.2,
    random_state=0
)

model1 = AdaBoost(T=50)
model1.fit(X_train, y_train)

model2 = AdaBoostClipped(T=50, cap=0.01)
model2.fit(X_train, y_train)

print("\nMax weight (vanilla):", max(model1.max_weights))
print("Max weight (clipped):", max(model2.max_weights))