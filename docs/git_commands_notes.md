# Git Commands – Notes

> Source: *Chapter-01 – Introduction to Git* (Advance Programming for Data Science, ITC – Mr. VANN Visal)

## 1. What is Git?
- Distributed version control system (created by Linus Torvalds, 2005).
- Tracks changes in source code (or any set of files); helps coordinate work among programmers.

## 2. Key Concepts
| Term | Meaning |
|------|---------|
| Repository (Repo) | Database storing all files and history of changes |
| Commit | Snapshot of the repository at a point in time |
| Branch | Separate line of development |
| Merge | Combine changes from different branches |
| Clone | Copy a repository to your local machine |
| Push / Pull | Sync changes between local and remote repositories |

---

## 3. Setting Up Git
Install from <https://git-scm.com/>, then configure your identity:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 4. Creating a New Repository
```bash
cd /path/to/your/project   # go to project directory
git init                   # initialize a new repository
```

## 5. Cloning an Existing Repository
```bash
git clone https://github.com/user/repo.git
```
Creates a directory `repo` containing all files and history.

## 6. Checking the Status
```bash
git status
```
Shows which files are staged, unstaged, or untracked.

## 7. Adding Files to the Staging Area
```bash
git add filename                  # one file
git add filename_1 filename_2     # several files
git add .                         # all files
git add *.extension               # all files of the same type
```

## 8. Renaming & Moving Files
```bash
git mv old_filename new_filename          # rename
git mv filename path/to/new_directory/    # move
```

## 9. Deleting Files
```bash
git rm filename            # delete a file from the repository
git rm -f filename         # force remove from working directory and index
git rm -rf directoryname   # force remove a directory and its contents
```
> ⚠️ Use with caution: files are removed from the working directory and staging area.

## 10. Committing Changes
```bash
git commit -m "Your commit message"   # commit staged changes
git commit --amend                    # change the last commit's message
```

## 11. Viewing Commit History
```bash
git log             # commits with messages, authors, dates
git log --oneline   # short one-line list
```

---

## 12–13. Branching
```bash
git checkout -b new-branch        # create a new branch (and switch to it)
git checkout existing-branch      # switch to an existing branch
```

## 14. Switching Branches (`git switch`)
Alternative to `git checkout` for changing branches.
```bash
git switch branch-name        # switch to an existing branch
git switch -c new-branch-name # create and switch to a new branch
```

## 15. Making Another Branch the Main Branch
```bash
git branch -m main old-main            # rename current main
git branch -m feature-branch main      # rename desired branch to main
git push origin -u main                # update the remote
git push origin --delete old-main      # delete old main from remote
```
> If `main` is protected, change the default branch in the repository settings first.

## 16. Merging Branches
```bash
git merge branch-to-merge     # merge into the current branch
```
Resolve any conflicts that arise.

## 17. Deleting Branches
```bash
git branch -d branch-to-delete             # delete locally
git push origin --delete branch-to-delete  # delete on remote
```

## 18. Syncing with Remote Repository
```bash
git push origin branch-name   # push changes to remote
git pull origin branch-name   # pull changes from remote
```

---

## Resolving Conflicts
- Conflicts occur when changes in different branches are incompatible.
- Git marks the conflicts in the affected files.
- Edit the files to resolve them, then `git add` and `git commit`.

## 19. Reverting Changes
```bash
git checkout commit-id -- filename   # revert a file to a previous commit
git reset --hard commit-id           # revert the entire repo to a previous commit
git revert commit-id                 # new commit that undoes the given commit
```
`git revert` creates a new commit with the inverse changes – undoes changes without altering history.

## 20. Stashing Changes
```bash
git stash                          # stash changes without committing
git stash push -m "Your message"   # stash with a message
git stash list                     # view stashes
git stash apply                    # apply latest stash
git stash apply index-stash        # apply a specific stash
git stash drop                     # drop latest stash
git stash drop index-stash         # drop a specific stash
git stash pop index-stash          # apply and remove a stash (finalize)
git stash clear                    # clear the whole stash list
```

## 21. Using `.gitignore`
- Tells Git which files/patterns to ignore; create it in the repository root.
- Common uses: OS-generated files, log/temp files, build artifacts.
```bash
# Example for a Python project
curl -L -o .gitignore https://www.toptal.com/developers/gitignore/api/python
```

## 22. Getting Help
```bash
git --help              # general help
git <command> --help    # help for a specific command
git clone --help
git status --help
```

## 23. Common Problems & Solutions
| Problem | Solution |
|---------|----------|
| Merge conflicts (same place changed in two branches) | Edit conflicting files, then commit |
| Detached HEAD (checked out a commit, not a branch) | `git checkout branch-name` |
| Forgot a file in a commit | `git add forgotten-file` then `git commit --amend` |

## 24. Tips for Effective Git Usage
- Write meaningful commit messages; commit often with small changes.
- Use branches to isolate features and fixes.
- Pull regularly from the remote; rebase before merging for a clean history.
- Communicate with teammates; assign tasks that edit different files.

Typical daily flow:
```bash
git status
git add filename
git commit -m "Your commit's message"
git pull
# if the same file was changed -> merge in an editor (e.g. VS Code)
git add filename
git commit -m "Your commit's message"
git push
```

---

## Practice: Team Workflow
```bash
# 1. Clone
git clone https://github.com/user/repo.git
# 2. Create a feature branch
git switch -c feature-branch
# 3. Make changes and commit
git add .
git commit -m "Add new feature"
# 4. Push the branch
git push origin feature-branch
```
5. **Create a pull request** on GitHub: "Compare & pull request" → review changes → create PR.
6. **Review & merge**: teammates review; after approval merge into `main`.
7. **Update local main**:
```bash
git switch main
git pull origin main
```
Everyone works on their own branch → fewer conflicts and easier code reviews.

## Additional Resources
Git Guide Book · Learning Branching · Atlassian · W3Schools
