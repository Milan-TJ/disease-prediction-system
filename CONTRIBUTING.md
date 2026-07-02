# Contributing Guide — Team 2 Disease Prediction System

This doc is the source of truth for how we work in this repo. Read it once, then it's mostly muscle memory.

## 1. Branching

- `main` is always deployable. Nobody pushes to it directly — every change comes in through a Pull Request (PR).
- Branch off `main` for every task:

```bash
git checkout main
git pull origin main
git checkout -b feature/short-description
```

**Naming convention:**
| Prefix | Use for |
|---|---|
| `feature/xxx` | New functionality |
| `fix/xxx` | Bug fixes |
| `data/xxx` | Data prep / preprocessing changes |
| `model/xxx` | ML model training / tuning changes |
| `docs/xxx` | README, docs, comments only |

Example: `feature/recipe-recommendation-api`, `fix/bmi-calculation-bug`

## 2. Commits

Keep commits small and message-clear. Format:

```
<type>: <short description>

feat: add symptom input validation
fix: correct BMI calculation in prediction model
data: clean missing values in symptoms dataset
model: tune random forest hyperparameters
docs: update setup instructions in README
```

## 3. Before opening a PR

Sync with `main` first so conflicts are small and yours to resolve, not the reviewer's:

```bash
git checkout main
git pull origin main
git checkout feature/your-branch
git merge main
# resolve any conflicts locally, then:
git push origin feature/your-branch
```

## 4. Opening a Pull Request

- Push your branch, then open a PR on GitHub targeting `main`.
- Fill out the PR template (auto-loads).
- Request review from at least **1 teammate**.
- Do not merge your own PR without a review, even if it feels trivial.

## 5. Reviewing someone else's PR

- Pull their branch locally if you want to actually run it: `git fetch origin && git checkout feature/their-branch`
- Leave comments for anything unclear — questions are fine, not just corrections.
- Approve when it's good to merge. Use "Squash and merge" on GitHub to keep `main`'s history clean.

## 6. After merge

```bash
git checkout main
git pull origin main
git branch -d feature/your-branch   # delete local branch, it's done
```

## 7. Data & model files

Large datasets and trained model weights (`.h5`, `.pt`, `.pkl`, etc.) are **not** committed to git — they're in `.gitignore`. Share these via the team's Drive/Colab link (add link in README) instead. Only small sample data goes in `data/sample/`.

## 8. If you're stuck on a conflict

Don't force-push over someone else's work. Ping the team channel — a 2-minute conversation beats an hour of git archaeology.
