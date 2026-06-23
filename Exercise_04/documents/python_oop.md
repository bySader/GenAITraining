# Object-Oriented Programming in Python

## Author: Jane Smith
## Topic: Programming

Object-Oriented Programming (OOP) is a programming paradigm that organizes software around objects rather than functions and logic. Python fully supports OOP through classes and objects.

## Core Concepts

### Classes and Objects
A class is a blueprint for creating objects. An object is an instance of a class.

```python
class Car:
    def __init__(self, make: str, model: str, year: int):
        self.make = make
        self.model = model
        self.year = year

    def describe(self) -> str:
        return f"{self.year} {self.make} {self.model}"

my_car = Car("Toyota", "Camry", 2023)
print(my_car.describe())  # 2023 Toyota Camry
```

### The Four Pillars of OOP

#### 1. Encapsulation
Bundling data and methods that operate on the data into a single unit (class), and restricting direct access to some components.

```python
class BankAccount:
    def __init__(self, balance: float):
        self.__balance = balance  # private attribute

    def deposit(self, amount: float):
        self.__balance += amount

    def get_balance(self) -> float:
        return self.__balance
```

#### 2. Inheritance
A class can inherit attributes and methods from another class.

```python
class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"
```

#### 3. Polymorphism
Different classes can be used interchangeably if they share the same interface.

```python
animals = [Dog(), Cat()]
for animal in animals:
    print(animal.speak())  # Woof! then Meow!
```

#### 4. Abstraction
Hiding complex implementation details and showing only the essential features.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Circle(Shape):
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2
```

## Special Methods (Dunder Methods)
Python classes can define special methods to support built-in operations:

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
```
