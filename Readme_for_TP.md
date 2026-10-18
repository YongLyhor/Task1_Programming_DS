# TP1 – Summary of Lesson 1 (Week 1)

**Course:** Advance Programming for Data Science

| Name       | ID        | Group     |
|------------|-----------|-----------|
| Yong Lyhor | e20221004 | I5-AMS-B  |

---

## Part 1: Activity in Class

### 1. Create a GitHub repository with three branches
At the start of the class, the teacher asked every student to create a GitHub repository and set up three branches:

| Branch   | Purpose                                         |
|----------|-------------------------------------------------|
| `main`   | The main (default) branch of the repository     |
| `course` | Notes and material from the lessons             |
| `TP`     | Practical work (TP) to submit to the teacher    |

Commands used:

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/YongLyhor/Task1_Programming_DS.git
git push -u origin main

git checkout -b course      # create and switch to branch "course"
git push -u origin course

git checkout -b TP          # create and switch to branch "TP"
git push -u origin TP
```

### 2. Slide review on coding
The teacher used slides to review programming basics:
- Introduction
- Clean Code, PEP 8, and Debugging
- Tips and Tricks to Help Ensure Your Code Follows PEP 8
- Coding concepts: **class**, **object**, **function**, **method**, and **how to debug code**

### 3. Question and answer
The teacher picked students one by one and asked each of them one coding question.

**My question:** *What is a class?*

**My answer:**
> A **class** is a blueprint (template) for creating objects. It defines the **attributes** (data/variables) and the **methods** (functions) that the objects created from it will have. An **object** is an instance of a class.

Example:

```python
class Student:
    """A simple class that represents a student."""

    def __init__(self, name, student_id, group):
        # attributes
        self.name = name
        self.student_id = student_id
        self.group = group

    def introduce(self):
        # method
        return f"My name is {self.name}, ID {self.student_id}, group {self.group}."


# create an object (instance) of the class
me = Student("Yong Lyhor", "e20221004", "I5-AMS-B")
print(me.introduce())
```

---

## Part 2: Summary of Lesson 1

### 1. Introduction
In Data Science we write a lot of code to clean data, analyse it, and build models. Code is **read much more often than it is written**, so it has to be easy for us and for others to read, understand, and maintain. This lesson covers how to write readable Python code that follows the standard style guide (PEP 8) and how to find and fix bugs.

### 2. Clean Code, PEP 8, and Debugging

#### 2.1 Clean Code
Clean code is code that is **easy to read, understand, and change**. Main ideas:
- Use **meaningful names** for variables, functions, and classes (`total_price`, not `tp`).
- Each function should do **one thing** and stay short.
- Avoid repeating code (**DRY** – Don't Repeat Yourself).
- Write comments and docstrings to explain *why*, not *what* the code obviously does.
- Keep the code simple (**KISS** – Keep It Simple, Stupid).

#### 2.2 Review of basic concepts

| Concept      | Description | Example |
|--------------|-------------|---------|
| **Function** | A reusable block of code that performs a task, defined with `def`. | `def add(a, b): return a + b` |
| **Class**    | A blueprint for creating objects. | `class Student: ...` |
| **Object**   | An instance created from a class. | `me = Student(...)` |
| **Method**   | A function defined inside a class; it works on the object (`self`). | `me.introduce()` |
| **Attribute**| A variable that belongs to an object or class. | `self.name` |

#### 2.3 PEP 8 – Style Guide for Python Code
PEP 8 is the official guideline for how Python code should look. Key rules:

**Naming conventions**

| Type                 | Style                  | Example            |
|----------------------|------------------------|--------------------|
| Variable / function  | `lower_case_with_underscores` (snake_case) | `student_name`, `calculate_mean()` |
| Class                | `CapWords` (PascalCase) | `DataLoader`       |
| Constant             | `UPPER_CASE`           | `MAX_SIZE = 100`   |
| Module               | short, lowercase       | `utils.py`         |

**Code layout**
- Use **4 spaces** per indentation level (no tabs).
- Maximum line length: **79 characters** (72 for comments/docstrings).
- **2 blank lines** around top-level functions and classes; **1 blank line** between methods in a class.
- Put imports at the top of the file, one per line, grouped in order: standard library → third-party → local.

**Whitespace and expressions**
- One space around operators: `x = a + b`, not `x=a+b`.
- No space inside brackets: `my_list[1]`, not `my_list[ 1 ]`.
- No space around `=` for keyword arguments: `func(x=1)`.
- Compare to `None` with `is` / `is not`, not `==`.
- Use `if my_list:` instead of `if len(my_list) > 0:`.

**Bad vs. Good example**

```python
# Bad
def CalcAvg(L):
    s=0
    for i in L:s+=i
    return s/len(L)

# Good (PEP 8)
def calculate_average(numbers):
    """Return the average of a list of numbers."""
    total = sum(numbers)
    return total / len(numbers)
```

#### 2.4 Debugging
Debugging is the process of **finding and fixing errors (bugs)** in code.

Types of errors:
- **Syntax error** – the code breaks Python grammar (e.g. missing `:`).
- **Runtime error / exception** – the code crashes while running (e.g. `ZeroDivisionError`, `TypeError`, `KeyError`).
- **Logical error** – the code runs but gives the wrong result.

Debugging techniques:
1. **Read the error message / traceback** carefully – it shows the line and type of error.
2. Use **`print()`** to check the value of variables.
3. Use the Python debugger **`pdb`** or `breakpoint()` to pause the code and inspect it step by step.
4. Use the **debugger in the IDE** (VS Code / PyCharm): set breakpoints, step over, step into, watch variables.
5. Handle expected errors with **`try` / `except`**.

```python
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Error: cannot divide by zero")
        return None


x = 10
breakpoint()   # pause here and inspect variables
print(safe_divide(x, 0))
```

### 3. Tips and Tricks to Help Ensure Your Code Follows PEP 8

| Tool | Type | What it does | Command |
|------|------|--------------|---------|
| `pycodestyle` | Linter | Checks code against PEP 8 rules | `pycodestyle my_file.py` |
| `flake8` | Linter | PEP 8 check + finds bugs and unused imports/variables | `flake8 my_file.py` |
| `pylint` | Linter | Detailed analysis and a quality score | `pylint my_file.py` |
| `black` | Autoformatter | Automatically reformats code into a consistent style | `black my_file.py` |
| `autopep8` | Autoformatter | Automatically fixes PEP 8 problems | `autopep8 --in-place my_file.py` |
| `isort` | Import sorter | Sorts and groups imports | `isort my_file.py` |

Other tips:
- Install the **Python / Pylance / Flake8 / Black** extensions in VS Code to see style problems while typing.
- Turn on **format on save** in the editor.
- Use **pre-commit hooks** so code is checked automatically before each `git commit`.
- Write **docstrings** for every module, class, and function.

Install the tools:

```bash
pip install pycodestyle flake8 pylint black autopep8 isort
```

### 4. What I learned
- How to create a GitHub repository and work with multiple branches (`main`, `course`, `TP`).
- Writing clean, readable code is as important as writing code that works.
- PEP 8 gives one common style so that all Python code looks consistent.
- How to read error messages and use `print()`, `pdb`, and the IDE debugger to fix bugs.
- Linters and autoformatters can check and fix PEP 8 style automatically.
- The difference between a class, an object, a function, and a method.

---

## Reference
1. PEP 8 – Style Guide for Python Code: <https://peps.python.org/pep-0008/>
2. Real Python – How to Write Beautiful Python Code With PEP 8: <https://realpython.com/python-pep8/>
3. Python Documentation – Classes: <https://docs.python.org/3/tutorial/classes.html>
4. Python Documentation – `pdb`, The Python Debugger: <https://docs.python.org/3/library/pdb.html>
5. Robert C. Martin, *Clean Code: A Handbook of Agile Software Craftsmanship*, Prentice Hall, 2008.
6. Course slides – Advance Programming for Data Science, Lesson 1.
