def pair_gcd(a,b):
    if a == 0:
        return b
 
    return pair_gcd(b % a, a)
def pair_egcd(a: int, b: int) -> tuple[int, int, int]:

    if a == 0:
        return (0, 1, b)
    else:
        x1, y1, d = pair_egcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (x, y, d)
def gcd(*args: int) -> int:

    if len(args) < 2:
        raise ValueError("At least two arguments are required")
    
    current_gcd = args[0]
    for number in args[1:]:
        current_gcd = pair_gcd(current_gcd, number)
        
    return current_gcd

def pair_lcm(a,b):
    return a*b//pair_gcd(a,b)

def lcm(*args: int) -> int:
     
    max_num = 0
    for i in range(len(args)):
        if (max_num < args[i]):
            max_num = args[i]

    res = 1
 

    x = 2; 
    while (x <= max_num):
         

        indexes = []
        for j in range(len(args)):
            if (args[j] % x == 0):
                indexes.append(j)
 

        if (len(indexes) >= 2):
             

            for j in range(len(indexes)):
                args[indexes[j]] = int(args[indexes[j]] / x);
 
            res = res * x
        else:
            x += 1
 

    for i in range(len(args)):
        res = res * args[i]
 
    return res
 
def are_relatively_prime(a: int, b: int) -> bool:
    if pair_gcd(a,b)==1:
        return True
    else:
        return False
def mod_inv(a: int, n: int) -> int:


    x, y, gcd = pair_egcd(a, n)
    if gcd != 1:
        raise Exception("a and n are not coprime")
    else:
        return x % n
    
def crt(a: list[int], n: list[int]) -> int:

    if len(a) != len(n) or len(a) == 0:
        raise ValueError("Lists 'a' and 'n' must have the same nonzero length")

    # Product of all n[i]
    N = 1
    for ni in n:
        N *= ni

    # Applying CRT
    x = 0
    for ai, ni in zip(a, n):
        Ni = N // ni
        mi = mod_inv(Ni, ni)
        x += ai * Ni * mi

    return x % N
def is_quadratic_residue_prime(a: int, p: int) -> int:

    if p <= 1:
        raise ValueError("p must be a prime number greater than 1")

    # Check if a is coprime to p
    if a % p == 0:
        return 0

    # Use Euler's criterion
    legendre_symbol = pow(a, (p - 1) // 2, p)
    
    if legendre_symbol == p - 1:
        return -1
    else:
        return legendre_symbol
def legendre_symbol(a: int, p: int) -> int:

    if p <= 1:
        raise ValueError("p must be a prime number greater than 1")

    if a % p == 0:
        return 0

    legendre_symbol = pow(a, (p - 1) // 2, p)
    return legendre_symbol

def is_quadratic_residue_prime_power(a: int, p: int, e: int) -> int:

    if e < 1:
        raise ValueError("e must be greater than or equal to 1")

    if e == 1:
        return legendre_symbol(a, p)
    if a % (p ** e) == 0:
        return 0
    jacobi_symbol = legendre_symbol(a, p)
    if jacobi_symbol == 0:
        return 0
    elif jacobi_symbol == 1:
        return 1
    else:
        if e % 2 == 0:
            return is_quadratic_residue_prime_power(a, p, e - 1)
        else:
            return -1

        