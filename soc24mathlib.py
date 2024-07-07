import array as arr
from typing import List

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
def floor_sqrt(x: int) -> int:

    
    return int(x**0.5)

def is_perfect_power(n):
    exponent = 2
    while True:
        if 2 ** exponent > n: 
            return False
        lo = 2
        hi = lo
        while hi ** exponent <= n:
            hi *= 2
        while hi - lo > 1:
            middle = (lo + hi) // 2
            if middle ** exponent <= n:
                lo = middle
            else:
                hi = middle
        if lo ** exponent == n:
            return True
        exponent += 1

import random

def is_prime(n: int, k=40) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    
    # Write (n - 1) as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    def miller_rabin_test(a, d, n, r):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False
    
    # Test with k different bases
    for _ in range(k):
        a = random.randint(2, n - 2)
        if not miller_rabin_test(a, d, n, r):
            return False
    return True

def gen_prime(m: int) -> int:
    while True:
        p = random.randint(2, m)
        if is_prime(p):
            return p

def gen_k_bit_prime(k: int) -> int:
    while True:
        p = random.getrandbits(k)
        if p >= 2 ** (k - 1) and is_prime(p):
            return p

def factor(n: int) -> list[tuple[int, int]]:
    if n == 1:
        return []
    
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while (n % d) == 0:
            n //= d
            count += 1
        if count > 0:
            factors.append((d, count))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def euler_phi(n: int) -> int:
    if n == 1:
        return 1
    factors = factor(n)
    phi = n
    for (p, _) in factors:
        phi = phi * (p - 1) // p
    return phi
class QuotientPolynomialRing:
    def __init__(self, poly: list[int], pi_gen: list[int]) -> None:
        if not pi_gen or pi_gen[-1] != 1:
            raise ValueError("pi_generator must be non-empty and monic.")
        self.element = self._mod_reduce(poly, pi_gen)
        self.pi_generator = pi_gen

    def __repr__(self):
        return f"QuotientPolynomialRing({self.element}, {self.pi_generator})"

    @staticmethod
    def _mod_reduce(poly: list[int], pi_gen: list[int]) -> list[int]:
        deg_pi = len(pi_gen) - 1
        poly = poly[:]
        while len(poly) > len(pi_gen):
            coef = poly[-1]
            for i in range(len(pi_gen)):
                poly[len(poly) - len(pi_gen) + i] -= coef * pi_gen[i]
            while poly and poly[-1] == 0:
                poly.pop()
        return poly

    @staticmethod
    def _add_or_sub(poly1, poly2, pi_gen, op):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials have different pi_generators.")
        len_max = max(len(poly1.element), len(poly2.element))
        result = [0] * len_max
        for i in range(len(poly1.element)):
            result[i] += poly1.element[i]
        for i in range(len(poly2.element)):
            if op == "add":
                result[i] += poly2.element[i]
            elif op == "sub":
                result[i] -= poly2.element[i]
        return QuotientPolynomialRing(result, pi_gen)

    @staticmethod
    def Add(poly1, poly2):
        return QuotientPolynomialRing._add_or_sub(poly1, poly2, poly1.pi_generator, "add")

    @staticmethod
    def Sub(poly1, poly2):
        return QuotientPolynomialRing._add_or_sub(poly1, poly2, poly1.pi_generator, "sub")
    @staticmethod
    def reduce(poly: List[int], pi_gen: List[int]) -> List[int]:
        # Reduce poly modulo pi_gen
        while len(poly) >= len(pi_gen):
            if poly[-1] != 0:
                factor = poly[-1]
                for i in range(len(pi_gen)):
                    poly[-(i+1)] = (poly[-(i+1)] - factor * pi_gen[-(i+1)]) % (10**9 + 7)
            poly.pop()
        return poly


    @staticmethod
    def Mul(poly1: 'QuotientPolynomialRing', poly2: 'QuotientPolynomialRing') -> 'QuotientPolynomialRing':
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials must have the same pi_generator.")
        product = [0] * (len(poly1.element) + len(poly2.element) - 1)
        for i, coef1 in enumerate(poly1.element):
            for j, coef2 in enumerate(poly2.element):
                product[i + j] = (product[i + j] + coef1 * coef2) % (10**9 + 7)
        return QuotientPolynomialRing(QuotientPolynomialRing.reduce(product, poly1.pi_generator), poly1.pi_generator)
    @staticmethod
    def GCD(poly1, poly2):
        if poly1.pi_generator != poly2.pi_generator:
            raise ValueError("Polynomials have different pi_generators.")
        a = poly1.element
        b = poly2.element
        while b != [0]:
            a, b = b, QuotientPolynomialRing._mod_reduce(
                [a[i] - b[i] for i in range(min(len(a), len(b)))], poly1.pi_generator)
        return QuotientPolynomialRing(a, poly1.pi_generator)

    @staticmethod
    def Inv(poly):
        if poly.element == [0]:
            raise ValueError("Zero polynomial is not invertible.")
        a, b = poly.pi_generator, poly.element
        x0, x1 = [1], [0]
        y0, y1 = [0], [1]
        while b != [0]:
            q, r = QuotientPolynomialRing._divmod(a, b)
            a, b = b, r
            x0, x1 = x1, [x0[i] - q[i] * x1[i] for i in range(min(len(x0), len(q)))]
            y0, y1 = y1, [y0[i] - q[i] * y1[i] for i in range(min(len(y0), len(q)))]
        if a != [1]:
            raise ValueError("Polynomial is not invertible.")
        return QuotientPolynomialRing(x0, poly.pi_generator)

    @staticmethod
    def _divmod(dividend: List[int], divisor: List[int]) -> (List[int], List[int]):
        if not any(divisor):
            raise ZeroDivisionError("division by zero polynomial")
        
        quotient = [0] * (len(dividend) - len(divisor) + 1)
        remainder = dividend[:]
        
        lead_coef_inv = pow(divisor[-1], -1, 10**9 + 7)
        
        for i in range(len(quotient)):
            quotient[i] = (remainder[-1] * lead_coef_inv) % (10**9 + 7)
            
            for j in range(len(divisor)):
                remainder[i + j] = (remainder[i + j] - quotient[i] * divisor[j]) % (10**9 + 7)
            
            remainder.pop()
        
        while remainder and remainder[-1] == 0:
            remainder.pop()
        
        return quotient, remainder





# class QuotientPolynomialRing:
#     def __init__(self, poly: list[int], pi_gen: list[int]):
#         if not pi_gen :
#             raise ValueError("pi_gen must be non-empty and monic.")
#         self.element = poly
#         self.pi_generator = pi_gen
#         self.degree = len(pi_gen) - 1

#     @staticmethod
#     def add_polynomials(poly1, poly2):
#         max_len = max(len(poly1), len(poly2))
#         poly1 += [0] * (max_len - len(poly1))
#         poly2 += [0] * (max_len - len(poly2))
#         return [(poly1[i] + poly2[i]) for i in range(max_len)]
    
#     @staticmethod
#     def sub_polynomials(poly1, poly2):
#         max_len = max(len(poly1), len(poly2))
#         poly1 += [0] * (max_len - len(poly1))
#         poly2 += [0] * (max_len - len(poly2))
#         return [(poly1[i] - poly2[i]) for i in range(max_len)]

#     @staticmethod
#     def mul_polynomials(poly1, poly2):
#         result = [0] * (len(poly1) + len(poly2) - 1)
#         for i in range(len(poly1)):
#             for j in range(len(poly2)):
#                 result[i + j] += poly1[i] * poly2[j]
#         return result
    
#     def mod_polynomial(self, poly):
#         while len(poly) > self.degree:
#             if poly[-1] != 0:
#                 for i in range(self.degree + 1):
#                     poly[-(self.degree + 1) + i] -= poly[-1] * self.pi_generator[i]
#             poly.pop()
#         return poly

#     @staticmethod
#     def gcd_polynomials(poly1, poly2):
#         while poly2:
#             poly1, poly2 = poly2, QuotientPolynomialRing.mod_polynomial(poly1, poly2)
#         return poly1

#     @staticmethod
#     def GCD(poly1, poly2):
#         if poly1.pi_generator != poly2.pi_generator:
#             raise ValueError("Polynomials must have the same pi_generator.")
#         result = QuotientPolynomialRing.gcd_polynomials(poly1.element, poly2.element)
#         return QuotientPolynomialRing(result, poly1.pi_generator)


#     @staticmethod
#     def Add(poly1, poly2):
#         if poly1.pi_generator != poly2.pi_generator:
#             raise ValueError("Polynomials must have the same pi_generator.")
#         result = QuotientPolynomialRing.add_polynomials(poly1.element, poly2.element)
#         result = poly1.mod_polynomial(result)
#         return QuotientPolynomialRing(result, poly1.pi_generator)
#     @staticmethod
#     def Mul(poly1, poly2):
#         if poly1.pi_generator != poly2.pi_generator:
#             raise ValueError("Polynomials must have the same pi_generator.")
#         result = QuotientPolynomialRing.mul_polynomials(poly1.element, poly2.element)
#         result = poly1.mod_polynomial(result)
#         return QuotientPolynomialRing(result, poly1.pi_generator)
#     @staticmethod
#     def Sub(poly1, poly2):
#         if poly1.pi_generator != poly2.pi_generator:
#             raise ValueError("Polynomials must have the same pi_generator.")
#         result = QuotientPolynomialRing.sub_polynomials(poly1.element, poly2.element)
#         result = poly1.mod_polynomial(result)
#         return QuotientPolynomialRing(result, poly1.pi_generator)
#     @staticmethod
#     def Inv(poly):
#         raise NotImplementedError("Modular inverse calculation is complex and will be added later.")


        


         

def aks_test(n: int) -> bool:
    c = [0] * (n+1)

    def coef(n):
        c[0] = 1
        for i in range(n):
            c[1 + i] = 1
            for j in range(i, 0, -1):
                c[j] = c[j - 1] - c[j]
            c[0] = -c[0]
     

    coef(n)

    c[0] = c[0] + 1
    c[n] = c[n] - 1
     

    i = n
    while (i > -1 and c[i] % n == 0):
        i = i - 1     

    return True if i < 0 else False
def log(x, b=2):
    if x < b:
        return 0  
    return 1 + log(x/b, b)
def floor(x):
    return int(x)

def findR(n):

   maxK = log(n)**2   
   maxR = log(n)**5   
   nexR = True              
   r = 1                   
   while nexR == True:
       r +=1
       nexR = False
       k = 0
       while k <= maxK and nexR == False:
           k = k+1
           if fastMod(n,k,r) == 0 or fastMod(n,k,r) == 1:
               nexR = True
   return(r)

def fastMod(base,power,n):

    r=1
    while power > 0:
        if power % 2 == 1:
            r = r * base % n
        base = base**2 % n
        power = power // 2
    return(r)

def fastPoly(base,power,r):

    x = arr.array('d',[],)
    a = base[0]

    for i in range(len(base)):
        x.append(0)
    x[(0)] = 1 
    n = power
    
    while power > 0:
        if power % 2 == 1: 
            x = multi(x,base,n,r)
        base = multi(base,base,n,r)
        power = power // 2

    x[(0)] = x[(0)] - a
    x[(n % r )] = x[(n % r )] - 1        
    return(x)

def multi(a,b,n,r):

    x = arr.array('d',[])
    for i in range(len(a)+len(b)-1):
        x.append(0)
    for i in range(len(a)):
        for j in range(len(b)):
            x[(i+j) % r ] += a[(i)] * b[(j)] 
            x[(i+j) % r] = x[(i+j) % r] % n 
    for i in range(r,len(x)):
            x=x[:-1]
    return(x)

def eulerPhi(r):

    x = 0        
    for i in range(1, r + 1):
        if pair_gcd(r, i) == 1:
            x += 1
    return x


def aks(n):

    if is_perfect_power(n) == True:                    
        return False
    
    r = findR(n)                                  

    for a in range(2,min(r,n)):                     
            return False

    if n <= r:                                    
        return True

    x = arr.array('l',[],)                        
    for a in range(1,floor((eulerPhi(r))**(1/2)*log(n))):      
        x = fastPoly(arr.array('l',[a,1]),n,r)
        if  any(x):
            return False
    return True  



        