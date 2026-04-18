from data import prepare_data
from adaboost import AdaBoost
from modifications import AdaBoostClipped, AdaBoostPersistent, AdaBoostSoft

for noise in [0.0, 0.1, 0.2, 0.25, 0.275, 0.3, 0.35]:
    X_train, X_test, y_train, y_test = prepare_data(
        dataset="moons",
        noise_level=noise
    )

    # Vanilla
    model1 = AdaBoost(T=20)
    model1.fit(X_train, y_train)
    acc1 = (model1.predict(X_test) == y_test).mean()

    # Clipped
    model2 = AdaBoostClipped(T=20, cap=0.01)
    model2.fit(X_train, y_train)
    acc2 = (model2.predict(X_test) == y_test).mean()

    # Persistent
    model3 = AdaBoostPersistent(T=20)
    model3.fit(X_train, y_train)
    acc3 = (model3.predict(X_test) == y_test).mean()
    model4 = AdaBoostSoft(T=20, beta=0.5)
    model4.fit(X_train, y_train)
    acc4 = (model4.predict(X_test) == y_test).mean()

    print(f"\nnoise={noise}")
    print(f"Vanilla:    {acc1:.4f}")
    print(f"Clipped:    {acc2:.4f}")
    print(f"Persistent: {acc3:.4f}")
    print(f"Soft:       {acc4:.4f}")