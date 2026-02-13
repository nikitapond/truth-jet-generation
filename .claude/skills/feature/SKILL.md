---
name: feature
description: End-to-end workflow for creating a new feature on a dedicated branch with planning, implementation, tests, and commit.
argument-hint: <feature-description>
disable-model-invocation: true
---

# Feature Creation Workflow

You are implementing a new feature described by: **$ARGUMENTS**

Follow these steps **in order**, completing each before moving to the next. If any step fails, stop and ask the user how to proceed.

---

## Step 1: Ensure clean working tree

Run `git status` and verify there are no uncommitted changes (staged or unstaged) and no untracked files that would be lost. If the working tree is not clean, **stop** and tell the user to commit or stash their changes before continuing.

## Step 2: Create a feature branch

- Identify the current branch (this is the base branch).
- Derive a short, descriptive branch name from the feature description using the convention `feature/<kebab-case-name>` (e.g. `feature/pythia-card-support`).
- Create and check out the new branch: `git checkout -b feature/<name>`

## Step 3: Plan the feature

Enter plan mode and thoroughly investigate the codebase before writing any code:

- Identify which files need to change and which new files (if any) are needed.
- Note existing patterns, conventions, and tests so the implementation is consistent.
- Write a clear, step-by-step implementation plan.
- Present the plan to the user for approval before proceeding.

## Step 4: Implement the feature

Implement the approved plan:

- Follow existing code style and conventions.
- Keep changes minimal and focused on the feature.
- Do not add unnecessary abstractions, comments, or unrelated refactors.

## Step 5: Create tests

- Add or update tests that cover the new functionality.
- Follow the existing test patterns in the `tests/` directory.
- Run `pytest tests/` to verify all tests pass (both new and existing).
- If any test fails, fix the issue and re-run until the suite is green.

## Step 6: Update documentation

- Update `CLAUDE.md` to reflect the new feature: add new CLI commands/flags to the Commands section, update Architecture descriptions for any changed or new modules, and note new config fields or output format changes where relevant.
- Keep edits concise and consistent with the existing doc style.

## Step 7: Commit the changes

- Run `git status` and `git diff` to review all changes.
- Stage only the files relevant to this feature, including doc updates (do not use `git add -A`).
- Write a concise commit message summarising the feature.
- Create the commit.

## Step 8: Prompt the user to push

Tell the user the feature branch name and suggest they push with:

```
git push -u origin feature/<name>
```

Do **not** push automatically.
