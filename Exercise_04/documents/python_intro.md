# Introduction to Python Programming

## Author: Jane Smith
## Topic: Programming

Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability and simplicity, allowing developers to express concepts in fewer lines of code than languages like C++ or Java.

## Key Features

- **Simple Syntax**: Python's syntax is clean and easy to read, making it ideal for beginners.
- **Interpreted**: Code runs line-by-line, making debugging easier.
- **Dynamically Typed**: Variables do not need type declarations.
- **Extensive Standard Library**: Python ships with a large collection of built-in modules.
- **Cross-platform**: Python runs on Windows, macOS, Linux, and more.

## Basic Data Types

Python has several built-in data types:

- `int` – Integer numbers (e.g., `42`)
- `float` – Floating-point numbers (e.g., `3.14`)
- `str` – Text strings (e.g., `"Hello, World!"`)
- `bool` – Boolean values (`True` or `False`)
- `list` – Ordered, mutable collections (e.g., `[1, 2, 3]`)
- `dict` – Key-value pairs (e.g., `{"name": "Alice"}`)
- `tuple` – Ordered, immutable collections (e.g., `(1, 2, 3)`)

## Control Flow

Python supports standard control flow statements:

```python
# if-else
x = 10
if x > 5:
    print("Greater than 5")
else:
    print("5 or less")

# for loop
for i in range(5):
    print(i)

# while loop
count = 0
while count < 3:
    count += 1
```

## Functions

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

result = greet("World")
print(result)  # Hello, World!
```

## Why Python?

Python is widely used in data science, machine learning, web development (Django, Flask), automation, and scripting. Its large community and rich ecosystem of third-party packages (NumPy, Pandas, TensorFlow, etc.) make it one of the most popular languages in the world.
