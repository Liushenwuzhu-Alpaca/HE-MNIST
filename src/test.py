def calculate_polyval(x: float) -> float:
    """多项式求值"""
    coeffs = [
        2.3139697111e-10,  # x^7
        1.1551724027,  # x^6
        -3.4105118408e-10,  # x^5
        -2.0891847536,  # x^4
        1.0966062321e-10,  # x^3
        1.4340123453,  # x^2
        0.4999999999,  # x^1
        0.0229645346,  # x^0
    ]

    result = coeffs[0]
    for i in range(1, len(coeffs)):
        result = result * x + coeffs[i]
    return result


if __name__ == "__main__":
    x = float(input())
    print(calculate_polyval(x))
