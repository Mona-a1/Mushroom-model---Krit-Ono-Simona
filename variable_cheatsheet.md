# 🍄 Mushroom Classification — Variable Cheat-Sheet

Hand-off note for the team. Use these variables to build more models without re-reading the whole notebook.

## 1. The key thing for modeling
Models take the **fully preprocessed matrix** (scaled + one-hot, **no NaN**) plus the target:
```python
model.fit(X_train_prep, y_train)
model.predict(X_test_prep)
model.score(X_test_prep, y_test)
```

| Variable | What it holds | Old name |
|---|---|---|
| `X_train_prep` / `X_test_prep` | model-ready: StandardScaler numbers + one-hot categories, no NaN | old `X_train`/`X_test` (but `get_dummies` + **unscaled**) |
| `X_train_robprep` / `X_test_robprep` | same, but with **RobustScaler** (only if you want to test robust) | (new) |
| `y_train` / `y_test` | target: 0 = edible, 1 = poisonous | same (source was `ys`) |

> Default = `X_train_prep` (Standard ≈ Robust; Standard marginally ahead).

## 2. ⚠️ Gotcha — `X_train` now means something different!
| Variable | What it holds NOW | Old name |
|---|---|---|
| `X` / `y` | raw features / target from the original `df` | used to be `Xs` / `ys` |
| `X_train` / `X_test` | **RAW**: categories as text + NaN (we split **before** encoding) | same name, new content |

> Old code that fed `X_train` straight into a model **now breaks** (strings + NaN). Replace `X_train` → `X_train_prep`, `X_test` → `X_test_prep`.

## 3. For EDA (readable, training data only)
| Variable | What it holds | Old name |
|---|---|---|
| `df_train` | training rows: readable categories + `class` + NaN → patterns / plots | (new) |
| `df_metr_nan` | numbers `0/1/2` only, with NaN | `df_metrical` |

> Do EDA **only on `df_train`**. Never on `df` (contains test) or `X_test` (the vault).

## 4. Intermediate steps (if you build your own preprocessing)
| Variable | What it holds |
|---|---|
| `num_colum` / `categ_colum` | column lists: `["0","1","2"]` / the 5 categorical columns |
| `X_train_median` / `X_test_median` | numbers after median-impute (still **un**scaled) |
| `X_train_scaled` / `X_train_robscaled` | numbers impute + Standard / Robust scaler |
| `X_train_onehot` / `X_test_onehot` | categories (imputed "MISSING") one-hot encoded |

How the model matrix is built: `X_train_prep = np.hstack([X_train_scaled, X_train_onehot])`

## 5. Fitted transformers (to treat NEW data the same way)
`scaler` (StandardScaler), `robscaler` (RobustScaler), `onehot` (OneHotEncoder).

> Note: `imputer` is used twice (median for numbers, then "MISSING" for categories) — it ends up pointing at the **category** imputer.

## 6. Models already built
`logreg` (LogReg), `knn` (kNN + StandardScaler), `knn2` (kNN + RobustScaler).
