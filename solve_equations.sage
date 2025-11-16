# Solve equations with Sage

var('x y')

# 1) Solve quadratic x^2 - 3*x + 2 == 0
sol1 = solve(x^2 - 3*x + 2 == 0, x)

# 2) Solve linear system x + y == 5, x - y == 1
sol2 = solve([x + y == 5, x - y == 1], [x, y])

# 3) Symbolic solving x^2 - 4 == 0
sol3 = solve(x^2 - 4 == 0, x)

print("Quadratic solutions:", sol1)
print("Linear system solutions:", sol2)
print("Symbolic solutions:", sol3)