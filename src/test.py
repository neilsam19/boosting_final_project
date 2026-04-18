from data import prepare_data
from adaboost import AdaBoost
from modifications import AdaBoostClipped

for noise in [0.4, 0.5]:
    X_train, X_test, y_train, y_test = prepare_data(
        dataset="moons",
        noise_level=noise
    )
    model1 = AdaBoost(T=20)
    model1.fit(X_train, y_train)

    acc1 = (model1.predict(X_test) == y_test).mean()
    print(f"Vanilla: {noise} {acc1}")

    model2 = AdaBoostClipped(T=20, cap=0.01)
    model2.fit(X_train, y_train)

    acc2 = (model2.predict(X_test) == y_test).mean()
    print(f"Clipped: {noise} {acc2}")
