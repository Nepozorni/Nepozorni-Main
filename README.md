# Nepozorni | GUI attentivenes app

---

## Prerequisites

Ensure you have the following installed and configured.

* **Git**: [Download and Install Git](https://git-scm.com/downloads)
* **Visual Studio Code**: [Download Visual Studio Code](https://code.visualstudio.com/)
* Make sure you clone the repository.

 --- 

 ## Branching Guidelines:
 
* **Release Branch**: Used for releases, only merge stable commits here.
* **Development Branch**: Used for development, merge features here.

> **Note**: For each Jira Issue, create a new branch with the Incident ID for better traceability.

---

## Creating a new Jira branch:

Each feature should have it's own branch. Create a branch in VS Code using the GUI or command line.

### Step 1: Open Nepozorni-Main project in Visual Studio Code

1. Open **Visual Studio Code**
2. Open the **Nepozorni-Main** folder.

### Step 2: Fetch the latest changes

#### GUI 
1. Go to the **Source Control** panel (usually a branch icon in the Activity Bar).
2. Click the three-dot menu (`...`) and select **Pull** or **Fetch** to ensure your local repository is updated with the latest changes.

#### Command Line

Alternatively, open the **Terminal** and enter:

```bash
   git fetch
```

### Step 3: Checkout the Base Branch

Make sure to start from the correct base branch (`release` or `development`), depending on whether you’re working on production or testing changes.

#### GUI

1. Go to the **Source Control** panel.
2. Open the **Branches** menu and select **Checkout to...**
3. Choose either the `main` or `test` branch as your base.

#### Command Line

Alternatively, use the terminal:

```bash
git checkout release
# or
git checkout development
```

### Step 4: Create a New Branch

Now, create a new branch based on the Jira ticket.

#### GUI

1. In the **Source Control** panel, click on the **Branches** dropdown.
2. Choose **Create new branch from ...** and select either the `main` or `test` branch
3. Enter a descriptive branch name that includes the Jira ID, such as `SCRUM-123-brief-description`, and press **Enter**.
4. Confirm that you have switched to your new branch.

#### Command Line

Alternatively, in the Terminal, you can use:

```bash
git checkout -b SCRUM-123-brief-description main
# or
git checkout -b SCRUM-123-brief-description test
```

5. Push the new branch to GitHub to make it available for the team:

```bash
git push -u origin SCRUM-123-brief-description
```

---

## Commiting and Pushing Changes

After making changes, commit and push them to GitHub.

### Step 1: Stage, Commit and Push Changes

#### GUI

1. In the Source Control panel, stage your changes by clicking the `+` icon next to each file, or click Stage All Changes.
2. Enter a commit message in the Commit box and press `Ctrl + Enter` to commit.
3. Click the three-dot menu (`...`) and select Push to send your changes to GitLab.

#### Command Line

Alternatively, in the Terminal:

```bash
git add .
git commit -m "Add description of your changes"
git push
```
