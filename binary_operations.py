from constants import DEFAULT_MODULO
import math
import random

def add_mod(x, y, p=DEFAULT_MODULO):
    return (x + y) % p

def product_mod(x, y, p=DEFAULT_MODULO):
    return (x * y) % p

def subtract_mod(x, y, p=DEFAULT_MODULO):
    return (x - y) % p

def divide_mod(x, y, p=DEFAULT_MODULO):
    # True modular division x * y^-1 mod p, via Fermat's little theorem (p prime).
    # The previous implementation used integer division (x // y) % p, which is not
    # the modular-division task described in the grokking literature.
    return (x * pow(y, p - 2, p)) % p

def add_square_mod(x, y, p=DEFAULT_MODULO):
    return (x**2 + y**2) % p

def factorial(x,y, p=DEFAULT_MODULO):
    return math.factorial(x)%p

def random_map(x,y,p=DEFAULT_MODULO):
    return random.randint(0,p)